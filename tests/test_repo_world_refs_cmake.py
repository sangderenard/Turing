from src.compiler import repo_world_refs_cmake as m


def refs(path, text, *tracked):
    return m.references(path, text, frozenset(tracked))


def test_contract_constants():
    assert m.LANGUAGE == "cmake"
    assert m.SUFFIXES == (".cmake",)
    assert m.FILENAMES == ("CMakeLists.txt",)


def test_subdirectory():
    r = refs("CMakeLists.txt", "add_subdirectory(src/core)\nadd_subdirectory(gone)\n",
             "src/core/CMakeLists.txt")
    assert r == [
        (1, "cmake-subdirectory", "src/core", "src/core/CMakeLists.txt", ""),
        (2, "cmake-subdirectory", "gone", None, "not tracked"),
    ]


def test_subdirectory_relative_to_file_dir_and_escape():
    r = refs("a/b/CMakeLists.txt", "add_subdirectory(../c)\nadd_subdirectory(../../../x)",
             "a/c/CMakeLists.txt")
    assert r[0][3] == "a/c/CMakeLists.txt"
    assert r[1][3] is None and "escapes" in r[1][4]


def test_include_path_vs_module():
    text = "include(cmake/auto_sources.cmake)\ninclude(CTest)\ninclude(GNUInstallDirs)\ninclude(foo.cmake)\n"
    r = refs("CMakeLists.txt", text, "cmake/auto_sources.cmake")
    assert r == [
        (1, "cmake-include", "cmake/auto_sources.cmake", "cmake/auto_sources.cmake", ""),
        (4, "cmake-include", "foo.cmake", None, "not tracked"),
    ]


def test_sources_multiline_lines_and_keywords():
    text = (
        "add_library(core SHARED\n"
        "  src/a.cpp\n"
        "  # a comment\n"
        "  src/b.cpp  # trailing\n"
        "  PRIVATE src/missing.cpp\n"
        ")\n"
    )
    r = refs("CMakeLists.txt", text, "src/a.cpp", "src/b.cpp")
    assert r == [
        (2, "cmake-source", "src/a.cpp", "src/a.cpp", ""),
        (4, "cmake-source", "src/b.cpp", "src/b.cpp", ""),
        (5, "cmake-source", "src/missing.cpp", None, "not tracked"),
    ]


def test_target_name_and_keywords_not_files():
    text = "add_executable(app WIN32 EXCLUDE_FROM_ALL main.cpp)\nadd_library(x OBJECT INTERFACE MODULE STATIC)\n"
    r = refs("CMakeLists.txt", text, "main.cpp")
    assert r == [(1, "cmake-source", "main.cpp", "main.cpp", "")]


def test_alias_skipped():
    assert refs("CMakeLists.txt", "add_library(A::B ALIAS ab.cpp)", "ab.cpp") == []


def test_target_sources():
    text = "target_sources(t PRIVATE a.cpp\n PUBLIC FILE_SET HEADERS BASE_DIRS inc FILES inc/x.h)"
    r = refs("d/CMakeLists.txt", text, "d/a.cpp", "d/inc/x.h")
    assert [(x[0], x[3]) for x in r] == [(1, "d/a.cpp"), (2, "d/inc/x.h")]


def test_set_and_list_append_variables():
    text = (
        "set(CORE_SOURCES a.cpp\n b.cpp)\n"
        "list(APPEND CORE_HEADERS x.h)\n"
        "set(OTHER a.cpp)\n"
        "list(APPEND OTHER b.cpp)\n"
        "set(MY_SRCS c.cpp)\n"
    )
    r = refs("CMakeLists.txt", text, "a.cpp", "b.cpp", "x.h", "c.cpp")
    assert [(x[0], x[2]) for x in r] == [(1, "a.cpp"), (2, "b.cpp"), (3, "x.h"), (6, "c.cpp")]


def test_quoted_and_bracket_arguments():
    text = 'add_library(t "src/a b.cpp" [[src/c.cpp]] [==[src/d.cpp]==])\n'
    r = refs("CMakeLists.txt", text, "src/a b.cpp", "src/c.cpp", "src/d.cpp")
    assert [x[3] for x in r] == ["src/a b.cpp", "src/c.cpp", "src/d.cpp"]


def test_multiline_quoted_argument_advances_line():
    text = 'message("one\ntwo")\nadd_library(t a.cpp)\n'
    assert refs("CMakeLists.txt", text, "a.cpp")[0][0] == 3


def test_comments_line_and_bracket():
    text = (
        "# add_library(t hidden.cpp)\n"
        "#[[ add_library(t hidden2.cpp)\n still comment ]]\n"
        "add_library(t real.cpp)\n"
    )
    r = refs("CMakeLists.txt", text, "real.cpp", "hidden.cpp", "hidden2.cpp")
    assert r == [(4, "cmake-source", "real.cpp", "real.cpp", "")]


def test_variable_substitution_rules():
    text = (
        "add_library(t ${CMAKE_CURRENT_SOURCE_DIR}/a.cpp ${CMAKE_CURRENT_LIST_DIR}/b.cpp\n"
        "  ${CMAKE_SOURCE_DIR}/top.cpp ${PROJECT_SOURCE_DIR}/src/c.cpp\n"
        "  ${OTHER}/d.cpp ${SRCS} @X@/e.cpp)\n"
    )
    r = refs("sub/CMakeLists.txt", text, "sub/a.cpp", "sub/b.cpp", "top.cpp", "src/c.cpp",
             "d.cpp", "e.cpp")
    assert [x[3] for x in r] == ["sub/a.cpp", "sub/b.cpp", "top.cpp", "src/c.cpp"]


def test_root_file_current_dir_variable():
    r = refs("CMakeLists.txt", "add_library(t ${CMAKE_CURRENT_SOURCE_DIR}/a.cpp)", "a.cpp")
    assert r[0][3] == "a.cpp"


def test_generator_expressions_skipped():
    text = "add_library(t $<$<CONFIG:Debug>:dbg.cpp> a.cpp)\ntarget_sources(t PRIVATE $<TARGET_OBJECTS:o>)"
    r = refs("CMakeLists.txt", text, "a.cpp", "dbg.cpp")
    assert [x[3] for x in r] == ["a.cpp"]


def test_configure_file():
    r = refs("CMakeLists.txt", "configure_file(cfg.h.in ${CMAKE_BINARY_DIR}/cfg.h @ONLY)", "cfg.h.in")
    assert r == [(1, "cmake-configure", "cfg.h.in", "cfg.h.in", "")]
    r = refs("CMakeLists.txt", "configure_file(nope.in out.h)")
    assert r == [(1, "cmake-configure", "nope.in", None, "not tracked")]


def test_glob_not_expanded():
    text = (
        "file(GLOB_RECURSE SRCS CONFIGURE_DEPENDS\n"
        "  \"${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp\"\n"
        "  RELATIVE foo\n"
        "  src/*.c)\n"
        "file(READ x.txt y)\n"
    )
    r = refs("CMakeLists.txt", text, "src/a.cpp")
    assert r == [
        (2, "cmake-glob", "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp", None, "glob not expanded"),
        (4, "cmake-glob", "src/*.c", None, "glob not expanded"),
    ]


def test_include_dirs_not_emitted():
    assert refs("CMakeLists.txt", "target_include_directories(t PUBLIC include)", "include/a.h") == []


def test_nodus_like_excerpt():
    text = (
        "cmake_minimum_required(VERSION 3.20)\n"
        "add_library(nodus_tensor_core SHARED\n"
        "  src/common/tensors/abstraction/tensor_registry.cpp\n"
        "  # Device execution for assembled modules\n"
        "  src/kernels/kernel_vulkan.cpp\n"
        ")\n"
        "add_library(canvas_tables SHARED\n"
        "  # placeholder (with parens) to keep syntax valid\n"
        ")\n"
        "add_library(Torch::Torch ALIAS torch)\n"
        "include(cmake/auto_sources.cmake)\n"
    )
    r = refs("CMakeLists.txt", text, "src/common/tensors/abstraction/tensor_registry.cpp",
             "src/kernels/kernel_vulkan.cpp", "cmake/auto_sources.cmake")
    assert [(x[0], x[1]) for x in r] == [(3, "cmake-source"), (5, "cmake-source"), (11, "cmake-include")]
    assert all(x[3] for x in r)


def test_nodus_auto_sources_glob():
    text = (
        "file(GLOB_RECURSE NODUS_CANVAS_SOURCES CONFIGURE_DEPENDS\n"
        '  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"\n'
        '  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"\n)\n'
        'list(FILTER NODUS_CANVAS_SOURCES EXCLUDE REGEX "[/\\\\]src[/\\\\]runtime[/\\\\]")\n'
    )
    r = refs("cmake/auto_sources.cmake", text)
    assert [(x[0], x[1]) for x in r] == [(2, "cmake-glob"), (3, "cmake-glob")]


def test_garbage_does_not_raise():
    for text in ("add_library(", "add_library(t (((", ")))", "((", 'add_library(t "unterminated',
                 "#[[ unterminated", "add_library(t [==[ unterminated", "\x00\xff(", "",
                 "include(", "file(GLOB", "set(", "list(APPEND"):
        assert isinstance(refs("CMakeLists.txt", text, "t"), list)


def test_unbalanced_returns_what_precedes():
    r = refs("CMakeLists.txt", "add_library(a x.cpp)\nadd_library(b y.cpp", "x.cpp", "y.cpp")
    assert r[0][3] == "x.cpp"


def test_deterministic():
    text = "add_library(t a.cpp b.cpp)\ninclude(x/y.cmake)\n"
    assert refs("CMakeLists.txt", text, "a.cpp") == refs("CMakeLists.txt", text, "a.cpp")
