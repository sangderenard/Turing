from src.compiler import repo_world_refs_html as m

T = frozenset({
    "site/index.html", "site/app.js", "site/lib/util.mjs", "site/lib/index.js", "site/style.css",
    "site/a.png", "site/s.mp3", "site/v.mp4", "site/other.html", "site/frame.html", "top.js",
    "site/sw.js", "site/manifest.json", "site/fav.ico",
})
P = "site/index.html"


def run(text, path=P, tracked=T):
    return m.references(path, text, tracked)


def test_contract_constants():
    assert m.LANGUAGE == "html" and m.SUFFIXES == (".html", ".htm")


def test_script():
    assert run('<script src="app.js?v=1#x"></script>') == [(1, "html-script", "app.js", "site/app.js", "")]


def test_link_kinds_and_rel_filter():
    t = ('<link rel="stylesheet" href="style.css">\n<link rel="shortcut icon" href="fav.ico">\n'
         '<link rel="manifest" href="manifest.json">\n<link rel="modulepreload" href="app.js">\n'
         '<link rel="preconnect" href="x.css">')
    r = run(t)
    assert [x[0] for x in r] == [1, 2, 3, 4]
    assert all(x[1] == "html-link" and x[3] for x in r)


def test_media():
    t = '<img src="a.png"><audio src="s.mp3"></audio><video src="v.mp4"></video><source src="a.png">'
    assert [x[1] for x in run(t)] == ["html-media"] * 4
    assert [x[3] for x in run(t)] == ["site/a.png", "site/s.mp3", "site/v.mp4", "site/a.png"]


def test_anchor_only_tracked_and_no_fragments():
    t = '<a href="#top">x</a><a href="other.html#s">y</a><a href="missing.html">z</a><a href="">e</a>'
    assert run(t) == [(1, "html-anchor", "other.html", "site/other.html", "")]


def test_frame():
    assert run('<iframe src="frame.html"></iframe>')[0][1:4] == ("html-frame", "frame.html", "site/frame.html")


def test_skip_urls():
    t = ('<script src="https://x/y.js"></script><script src="//cdn/y.js"></script>'
         '<img src="data:image/png;base64,AA"><iframe src="blob:abc"></iframe>'
         '<a href="javascript:void(0)">a</a><a href="mailto:a@b.c">m</a>')
    assert run(t) == []


def test_served_root_resolves_beside_page():
    assert run('<script src="/app.js"></script>')[0][3] == "site/app.js"


def test_served_root_not_beside_page():
    r = run('<script src="/top.js"></script>')
    assert r == [(1, "html-script", "/top.js", None, "served-root path; not beside the page")]


def test_relative_parent_and_escape_and_missing():
    assert run('<script src="../top.js"></script>')[0][3] == "top.js"
    r = run('<script src="../../x.js"></script>')
    assert r[0][3] is None and r[0][4]
    assert run('<script src="nope.js"></script>')[0][4] == "not tracked"


def test_inline_module_imports_and_lines():
    t = ('<html>\n<script type="module">\n'
         'import a from "./app.js";\n// import b from "./no.js"\n'
         'const s = "import(\'./x.js\')";\n'
         'const m = await import("./lib/util.mjs");\n'
         'import "./lib";\nexport * from "react";\n</script>')
    r = run(t)
    assert [(x[0], x[2], x[3]) for x in r] == [
        (3, "./app.js", "site/app.js"),
        (6, "./lib/util.mjs", "site/lib/util.mjs"),
        (7, "./lib", "site/lib/index.js"),
        (8, "react", None),
    ]
    assert all(x[1] == "html-inline-import" for x in r)
    assert r[3][4] == "bare specifier"


def test_inline_non_module_ignored():
    assert run('<script>import("./app.js")</script>') == []


def test_crlf():
    r = run('<p>\r\n<img src="a.png">\r\n<script src="app.js"></script>')
    assert [x[0] for x in r] == [2, 3]


def test_malformed_never_raises():
    for bad in ['<script src="a', '<a href=', '<<<>>>', '<script type="module">import(', '', "<img src='x"]:
        run(bad)


def test_deterministic():
    t = '<img src="a.png"><script src="app.js"></script>'
    assert run(t) == run(t)
