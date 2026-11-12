from src.compiler import repo_world_refs_js_dynamic as m

T = frozenset({
    "web/main.js", "web/w.js", "web/lib/index.js", "web/mod.mjs", "web/k.wasm", "web/d.json",
    "web/s.wgsl", "web/x.js", "top.js", "web/imp.js",
})
P = "web/main.js"


def run(text, path=P, tracked=T):
    return m.references(path, text, tracked)


def kinds(text):
    return [(r[1], r[2], r[3]) for r in run(text)]


def test_contract_constants():
    assert m.LANGUAGE == "javascript" and m.SUFFIXES == (".js", ".mjs", ".cjs")


def test_dynamic_import_suffix_branches():
    assert kinds('import("./x.js")') == [("js-dynamic-import", "./x.js", "web/x.js")]
    assert run('import("./mod")')[0][3] == "web/mod.mjs"
    assert run('import("./lib")')[0][3] == "web/lib/index.js"
    assert run('import("./w")')[0][3] == "web/w.js"
    assert run('import("../top")')[0][3] == "top.js"


def test_dynamic_import_failures():
    assert run('import("react")') == [(1, "js-dynamic-import", "react", None, "bare specifier")]
    assert run('import("./nope")')[0][4] == "not tracked"
    assert run('import("../../x")')[0][3] is None
    assert run('import("https://a/b.js")')[0][4] == "bare specifier"


def test_served_root():
    assert run('import("/x.js")')[0][3] == "web/x.js"
    assert run('import("/top.js")')[0][4] == "served-root path; not beside the page"


def test_worker_forms():
    t = ('new Worker("./w.js");\nnew SharedWorker("w.js");\n'
         'new Worker(new URL("./w.js", import.meta.url), {type:"module"});')
    r = run(t)
    assert [(x[0], x[1], x[3]) for x in r] == [(1, "js-worker", "web/w.js"), (2, "js-worker", "web/w.js"),
                                               (3, "js-worker", "web/w.js")]


def test_import_scripts():
    r = run('importScripts("./w.js", "./x.js");\nimportScripts("nope.js")')
    assert [(x[0], x[1], x[3]) for x in r] == [(1, "js-import-scripts", "web/w.js"),
                                               (1, "js-import-scripts", "web/x.js"),
                                               (2, "js-import-scripts", None)]


def test_url_not_double_counted():
    r = run('const u = new URL("./k.wasm", import.meta.url);')
    assert [(x[1], x[3]) for x in r] == [("js-url", "web/k.wasm")]
    assert len(run('new Worker(new URL("./w.js", import.meta.url))')) == 1
    assert run('new URL("./k.wasm", base)') == []


def test_require():
    assert kinds('const a = require("./imp.js")') == [("js-require", "./imp.js", "web/imp.js")]
    assert run('require("fs")')[0][4] == "bare specifier"
    assert run('obj.require("./imp.js")') == []


def test_fetch_asset():
    t = 'fetch("./k.wasm");fetch("d.json?x=1");fetch("./s.wgsl");fetch("./page.html");fetch("https://a/b.json")'
    r = run(t)
    assert [(x[1], x[3]) for x in r] == [("js-fetch-asset", "web/k.wasm"), ("js-fetch-asset", "web/d.json"),
                                         ("js-fetch-asset", "web/s.wgsl")]


def test_skip_template_and_nonliteral():
    assert run('import(`./x${n}.js`)') == []
    assert run('import(`./x.js`)') == []
    assert run('import(name)') == []
    assert run('new Worker(url)') == []


def test_comments_and_strings_avoided():
    t = ('// import("./x.js")\n/* new Worker("./w.js")\n require("./imp.js") */\n'
         'const s = "import(\\"./x.js\\")";\nconst t = \'fetch("./k.wasm")\';\n'
         'const r = /import\\("a"\\)/;\nimport("./w.js");')
    r = run(t)
    assert [(x[0], x[2]) for x in r] == [(7, "./w.js")]


def test_static_import_and_meta_not_matched():
    assert run('import x from "./x.js";\nimport "./w.js";\nconst u = import.meta.url;') == []


def test_line_numbers_across_comment_and_template():
    t = '/* a\nb */\nconst q = `a\n${1}`;\nrequire("./imp.js")'
    assert run(t)[0][0] == 5


def test_crlf():
    assert [x[0] for x in run('a\r\nb\r\nrequire("./imp.js")\r\n')] == [3]


def test_malformed_never_raises():
    for bad in ['import("', 'new Worker(', '/*', '`${', '"abc', "importScripts(", "/[", 'fetch("a.json', "\x00\ufffd"]:
        run(bad)


def test_deterministic():
    t = 'require("./imp.js");import("./x.js");new Worker("./w.js")'
    assert run(t) == run(t)
