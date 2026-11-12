from src.compiler.repo_world_refs_c import LANGUAGE, SUFFIXES, references

T = frozenset({
    "a/b/x.c", "a/b/y.h", "a/z.h", "inc/util.h", "p/common.h", "q/common.h",
    "lib/deep/only.h", "a/b/impl.inl", "top.h",
})


def refs(path, text, tracked=T):
    return references(path, text, tracked)


def test_meta():
    assert LANGUAGE == "c" and ".inl" in SUFFIXES


def test_relative():
    assert refs("a/b/x.c", '#include "y.h"\n') == [(1, "c-include", "y.h", "a/b/y.h", "")]


def test_parent_relative():
    assert refs("a/b/x.c", '#include "../z.h"')[0][3] == "a/z.h"


def test_parent_escape_not_tracked():
    assert refs("top.c", '#include "../z.h"')[0][3] is None


def test_root_exact():
    assert refs("a/b/x.c", '#  include "inc/util.h"')[0][3] == "inc/util.h"


def test_unique_suffix():
    r = refs("a/b/x.c", "#include <deep/only.h>")[0]
    assert r[3] == "lib/deep/only.h" and r[4] == ""


def test_ambiguous():
    r = refs("a/b/x.c", '#include "common.h"')[0]
    assert r[3] is None and r[4].startswith("ambiguous: 2 candidates") and "p/common.h" in r[4]


def test_system_header():
    r = refs("a/b/x.c", "#include <stdio.h>")[0]
    assert r[3] is None and r[4] == "not tracked"


def test_commented_out():
    text = '// #include "y.h"\n/* #include "y.h"\n#include "y.h"\n*/\n#include "y.h" // c\n'
    assert [r[0] for r in refs("a/b/x.c", text)] == [5]


def test_inside_string():
    text = 'const char *s = "#include <y.h>";\nchar *t = "/*";\n#include "y.h"\n'
    assert [r[0] for r in refs("a/b/x.c", text)] == [3]


def test_inl_includes_h_and_import():
    assert refs("a/b/impl.inl", '#import "y.h"')[0][3] == "a/b/y.h"


def test_crlf():
    r = refs("a/b/x.c", '#include "y.h"\r\n#include <q.h>\r\n')
    assert [x[0] for x in r] == [1, 2] and r[0][3] == "a/b/y.h"


def test_garbage_no_raise():
    for text in ['\x00\xff\ufffd#include "', '#include <', '/* unterminated', '"abc\\', "'", "#include", "#include \"\""]:
        assert isinstance(refs("a/b/x.c", text), list)


def test_determinism():
    text = '#include "y.h"\n#include "common.h"\n#include <stdio.h>\n'
    assert refs("a/b/x.c", text) == refs("a/b/x.c", text)
