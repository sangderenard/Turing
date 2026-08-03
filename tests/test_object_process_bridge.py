import pytest

from src.compiler.object_process_bridge import raise_object_method_to_process_graph


def test_selected_method_uses_semantic_process_graph_and_retains_identity():
    raised = raise_object_method_to_process_graph(
        """
class WordOps:
    @staticmethod
    def xor(x, y):
        return x ^ y
""",
        class_name="WordOps",
        method_name="xor",
        source_filename="word_ops.py",
    )

    operations = {
        payload["op"] for _node, payload in raised.process_graph.G.nodes(data=True)
    }
    assert {"input", "bitxor", "return"} <= operations
    assert "opaque_python" not in operations
    assert raised.identity.graph_identity == "WordOps.xor"
    assert raised.identity.decorators == ("staticmethod",)
    assert raised.identity.method_source_span[0] == 4
    assert raised.process_graph.G.graph["object_origin"] == {
        "class_name": "WordOps",
        "method_name": "xor",
        "graph_identity": "WordOps.xor",
        "class_source_span": raised.identity.class_source_span,
        "method_source_span": raised.identity.method_source_span,
        "decorators": ("staticmethod",),
        "source_filename": "word_ops.py",
    }


@pytest.mark.parametrize(
    ("class_name", "method_name", "message"),
    [
        ("Missing", "xor", "exactly one class"),
        ("WordOps", "missing", "exactly one method"),
    ],
)
def test_selected_method_reports_missing_identity(
    class_name: str,
    method_name: str,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        raise_object_method_to_process_graph(
            """
class WordOps:
    def xor(self, x, y):
        return x ^ y
""",
            class_name=class_name,
            method_name=method_name,
        )


def test_duplicate_source_identity_is_rejected_instead_of_guessed():
    with pytest.raises(ValueError, match="found 2"):
        raise_object_method_to_process_graph(
            """
class WordOps:
    def xor(self, x, y):
        return x ^ y

class WordOps:
    def xor(self, x, y):
        return y ^ x
""",
            class_name="WordOps",
            method_name="xor",
        )
