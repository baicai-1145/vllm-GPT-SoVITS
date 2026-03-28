#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <utility>
#include <tuple>
#include <vector>

namespace {

constexpr int kAmbiguousType = 0;
constexpr int kBridgeType = 1;
constexpr int kEnType = 2;
constexpr int kKoType = 3;
constexpr int kKanaType = 4;
constexpr int kHanType = 5;

using Span = std::tuple<Py_ssize_t, Py_ssize_t, int>;
using SpanRow = std::vector<Span>;

struct TextView {
    PyObject *obj;
    int kind;
    void *data;
    Py_ssize_t length;
};

inline bool is_unicode_bridge(Py_UCS4 cp) {
    return cp == 0x3001 || cp == 0xFF0C || cp == 0x3002 || cp == 0xFF01 || cp == 0xFF1F || cp == 0x2026 ||
           cp == 0xFF1A || cp == 0xFF1B || cp == 0x2014 || cp == 0xFF5E || cp == 0x002F || cp == 0x00B7;
}

inline int classify_codepoint(Py_UCS4 cp) {
    if ((cp >= '0' && cp <= '9') || Py_UNICODE_ISSPACE(cp) || cp == ',' || cp == '.' || cp == '!' || cp == '?' ||
        cp == '-' || cp == '/' || cp == '~' || is_unicode_bridge(cp)) {
        return kBridgeType;
    }
    if ((cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z') || (cp >= 0xFF21 && cp <= 0xFF3A) ||
        (cp >= 0xFF41 && cp <= 0xFF5A)) {
        return kEnType;
    }
    if ((cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F) || (cp >= 0xAC00 && cp <= 0xD7AF)) {
        return kKoType;
    }
    if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0xFF66 && cp <= 0xFF9D)) {
        return kKanaType;
    }
    if ((cp >= 0x4E00 && cp <= 0x9FFF) || cp == 0x3005) {
        return kHanType;
    }
    return kAmbiguousType;
}

inline bool is_en_run_core(Py_UCS4 cp, int char_type) {
    return char_type == kEnType || Py_UNICODE_ISDIGIT(cp) || cp == '\'' || cp == '_' || cp == '-';
}

PyObject *scan_selective_direct_runs(PyObject *, PyObject *args) {
    PyObject *texts_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &texts_obj)) {
        return nullptr;
    }

    PyObject *sequence = PySequence_Fast(texts_obj, "texts must be a sequence");
    if (sequence == nullptr) {
        return nullptr;
    }

    const Py_ssize_t text_count = PySequence_Fast_GET_SIZE(sequence);
    PyObject **items = PySequence_Fast_ITEMS(sequence);
    PyObject *rows = PyList_New(text_count);
    if (rows == nullptr) {
        Py_DECREF(sequence);
        return nullptr;
    }

    std::vector<TextView> views;
    views.reserve(static_cast<size_t>(text_count));
    for (Py_ssize_t text_index = 0; text_index < text_count; ++text_index) {
        PyObject *text_obj = items[text_index];
        if (!PyUnicode_Check(text_obj)) {
            PyObject *converted = PyObject_Str(text_obj);
            if (converted == nullptr) {
                Py_DECREF(rows);
                Py_DECREF(sequence);
                return nullptr;
            }
            text_obj = converted;
        } else {
            Py_INCREF(text_obj);
        }

        if (PyUnicode_READY(text_obj) != 0) {
            Py_DECREF(text_obj);
            Py_DECREF(rows);
            Py_DECREF(sequence);
            return nullptr;
        }
        views.push_back(
            TextView{
                text_obj,
                static_cast<int>(PyUnicode_KIND(text_obj)),
                PyUnicode_DATA(text_obj),
                PyUnicode_GET_LENGTH(text_obj),
            }
        );
    }

    bool has_direct_run = false;
    std::vector<SpanRow> all_spans(static_cast<size_t>(text_count));
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t text_index = 0; text_index < text_count; ++text_index) {
        const TextView &view = views[static_cast<size_t>(text_index)];
        const int kind = view.kind;
        void *data = view.data;
        const Py_ssize_t text_length = view.length;
        std::vector<int> char_types(static_cast<size_t>(text_length));
        for (Py_ssize_t index = 0; index < text_length; ++index) {
            const auto cp = PyUnicode_READ(kind, data, index);
            char_types[static_cast<size_t>(index)] = classify_codepoint(cp);
        }

        SpanRow spans;
        Py_ssize_t cursor = 0;
        Py_ssize_t pending_bridge_start = -1;

        while (cursor < text_length) {
            const int current_type = char_types[static_cast<size_t>(cursor)];
            if (current_type == kBridgeType) {
                if (pending_bridge_start < 0) {
                    pending_bridge_start = cursor;
                }
                ++cursor;
                continue;
            }

            const Py_ssize_t segment_start = pending_bridge_start >= 0 ? pending_bridge_start : cursor;
            pending_bridge_start = -1;

            if (current_type == kEnType) {
                ++cursor;
                while (cursor < text_length) {
                    const auto cp = PyUnicode_READ(kind, data, cursor);
                    const int next_type = char_types[static_cast<size_t>(cursor)];
                    if (is_en_run_core(cp, next_type)) {
                        ++cursor;
                        continue;
                    }

                    const Py_ssize_t bridge_start = cursor;
                    while (cursor < text_length && char_types[static_cast<size_t>(cursor)] == kBridgeType) {
                        ++cursor;
                    }
                    if (cursor >= text_length) {
                        cursor = bridge_start;
                        break;
                    }

                    const auto bridge_cp = PyUnicode_READ(kind, data, cursor);
                    if (!is_en_run_core(bridge_cp, char_types[static_cast<size_t>(cursor)])) {
                        cursor = bridge_start;
                        break;
                    }
                }
                spans.emplace_back(segment_start, cursor, kEnType);
                has_direct_run = true;
                continue;
            }

            if (current_type == kKoType) {
                ++cursor;
                while (cursor < text_length) {
                    if (char_types[static_cast<size_t>(cursor)] == kKoType) {
                        ++cursor;
                        continue;
                    }

                    const Py_ssize_t bridge_start = cursor;
                    while (cursor < text_length && char_types[static_cast<size_t>(cursor)] == kBridgeType) {
                        ++cursor;
                    }
                    if (cursor >= text_length) {
                        cursor = bridge_start;
                        break;
                    }
                    if (char_types[static_cast<size_t>(cursor)] != kKoType) {
                        cursor = bridge_start;
                        break;
                    }
                }
                spans.emplace_back(segment_start, cursor, kKoType);
                has_direct_run = true;
                continue;
            }

            if (current_type == kKanaType || current_type == kHanType) {
                bool saw_kana = current_type == kKanaType;
                bool saw_han = current_type == kHanType;
                ++cursor;
                Py_ssize_t last_core_end = cursor;
                while (cursor < text_length) {
                    const int next_type = char_types[static_cast<size_t>(cursor)];
                    if (next_type == kHanType) {
                        saw_han = true;
                        ++cursor;
                        last_core_end = cursor;
                        continue;
                    }
                    if (next_type == kKanaType) {
                        saw_kana = true;
                        ++cursor;
                        last_core_end = cursor;
                        continue;
                    }
                    if (next_type == kBridgeType) {
                        ++cursor;
                        continue;
                    }
                    break;
                }
                cursor = last_core_end;
                spans.emplace_back(segment_start, last_core_end, saw_kana && !saw_han ? kKanaType : kAmbiguousType);
                if (saw_kana && !saw_han) {
                    has_direct_run = true;
                }
                continue;
            }

            ++cursor;
            Py_ssize_t last_core_end = cursor;
            while (cursor < text_length) {
                const int next_type = char_types[static_cast<size_t>(cursor)];
                if (next_type != kAmbiguousType && next_type != kBridgeType) {
                    break;
                }
                ++cursor;
                if (next_type != kBridgeType) {
                    last_core_end = cursor;
                }
            }
            cursor = last_core_end;
            spans.emplace_back(segment_start, last_core_end, kAmbiguousType);
        }

        if (pending_bridge_start >= 0) {
            if (!spans.empty()) {
                std::get<1>(spans.back()) = text_length;
            } else {
                spans.emplace_back(pending_bridge_start, text_length, kAmbiguousType);
            }
        }
        all_spans[static_cast<size_t>(text_index)] = std::move(spans);
    }
    Py_END_ALLOW_THREADS

    for (Py_ssize_t text_index = 0; text_index < text_count; ++text_index) {
        const TextView &view = views[static_cast<size_t>(text_index)];
        const SpanRow &spans = all_spans[static_cast<size_t>(text_index)];
        PyObject *row = PyList_New(static_cast<Py_ssize_t>(spans.size()));
        if (row == nullptr) {
            for (const TextView &cleanup_view : views) {
                Py_DECREF(cleanup_view.obj);
            }
            Py_DECREF(rows);
            Py_DECREF(sequence);
            return nullptr;
        }
        for (Py_ssize_t span_index = 0; span_index < static_cast<Py_ssize_t>(spans.size()); ++span_index) {
            const auto &[start, end, lang_type] = spans[static_cast<size_t>(span_index)];
            PyObject *chunk = PyUnicode_Substring(view.obj, start, end);
            if (chunk == nullptr) {
                Py_DECREF(row);
                for (const TextView &cleanup_view : views) {
                    Py_DECREF(cleanup_view.obj);
                }
                Py_DECREF(rows);
                Py_DECREF(sequence);
                return nullptr;
            }
            PyObject *span = Py_BuildValue("(Oi)", chunk, lang_type);
            Py_DECREF(chunk);
            if (span == nullptr) {
                Py_DECREF(row);
                for (const TextView &cleanup_view : views) {
                    Py_DECREF(cleanup_view.obj);
                }
                Py_DECREF(rows);
                Py_DECREF(sequence);
                return nullptr;
            }
            PyList_SET_ITEM(row, span_index, span);
        }
        PyList_SET_ITEM(rows, text_index, row);
        Py_DECREF(view.obj);
    }

    Py_DECREF(sequence);
    if (!has_direct_run) {
        Py_DECREF(rows);
        Py_RETURN_NONE;
    }
    return rows;
}

PyMethodDef kMethods[] = {
    {"scan_selective_direct_runs", scan_selective_direct_runs, METH_VARARGS, nullptr},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModuleDef = {
    PyModuleDef_HEAD_INIT,
    "gptsovits_split_fastpath",
    nullptr,
    -1,
    kMethods,
};

}  // namespace

PyMODINIT_FUNC PyInit_gptsovits_split_fastpath(void) {
    return PyModule_Create(&kModuleDef);
}
