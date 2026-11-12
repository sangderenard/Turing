"""ast_control_normalization._filter_torch_device_placement: device placement leaves the source program until the
deployment section owns it (VRAM loaders and shaders for numeric sections)."""
import ast

from src.compiler.ast_control_normalization import _filter_torch_device_placement


def _filtered(source: str) -> tuple[str, tuple]:
    tree = ast.parse(source)
    receipts = _filter_torch_device_placement(tree)
    return ast.unparse(tree), receipts


def test_device_keywords_leave_tensor_operations_and_dtype_stays():
    text, receipts = _filtered(
        "offsets = AbstractTensor.tensor(values, dtype=self.precision, device=self.device)\n"
        "loaded = AbstractTensor.tensor(data).to(device=self.device, dtype=self.precision)\n"
        "z = AbstractTensor.zeros((4, 3), dtype='float64', device='cuda')\n"
    )
    assert "device" not in text
    assert "AbstractTensor.tensor(values, dtype=self.precision)" in text
    assert "AbstractTensor.tensor(data).to(dtype=self.precision)" in text
    assert "AbstractTensor.zeros((4, 3), dtype='float64')" in text
    assert len(receipts) == 3 and all(r["dropped_device_keyword"] for r in receipts)


def test_a_call_that_only_places_a_value_is_its_receiver():
    text, receipts = _filtered(
        "a = x.to(device=self.device)\n"
        "b = y.cpu()\n"
        "c = z.cuda()\n"
        "d = w.to_device(target)\n"
    )
    assert text.splitlines() == ["a = x", "b = y", "c = z", "d = w"]
    assert [r["placement_call_is_its_receiver"] for r in receipts] == [True] * 4


def test_what_is_not_placement_is_left_as_authored():
    source = (
        "a = x.to(self.precision)\n"          # positional: may be a dtype
        "b = x.to(dtype='float32')\n"
        "c = open_device(device='cuda')\n"    # not a tensor operation
        "d = x.cuda(1)\n"                     # an argument it does not understand
    )
    text, receipts = _filtered(source)
    assert text == ast.unparse(ast.parse(source)) and receipts == ()
