from pathlib import Path
import ast
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.compiler.extraction_contract import ExtractionContract
from src.transmogrifier.graph import graph_express2

root = Path(__file__).resolve().parents[3]
turing = root / "turing"
source_path = root / "spectral-analyzer" / "electrical_dt_engine.py"
sys.path.insert(0, str(source_path.parent))
contract = ExtractionContract(
    turing / "extraction_contracts" / "program_extraction.yaml"
).with_sources([
    ("electrical_dt_engine", source_path),
    ("electrical_tensor_network", root / "spectral-analyzer" / "electrical_tensor_network.py"),
    ("engine_toy.dc_power", turing / "engine_toy" / "dc_power.py"),
])
tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
tree, links, unresolved, bindings = graph_express2._expand_unresolved_ast_parents(
    tree, {}, include=contract, pursuit_roots=("ComplexElectricalEngine.step",),
)
definition = next(
    node for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "ComplexElectricalEngine"
)
initializer = next(
    node for node in definition.body
    if isinstance(node, ast.FunctionDef) and node.name == "__init__"
)
field_bindings = dict(getattr(definition, "_python_bindings", None) or bindings)
local_bindings = graph_express2._ast_local_constructor_bindings(
    initializer, field_bindings,
)
print("class_binding_keys=" + repr(sorted(field_bindings)), flush=True)
print("empirical_bindings=" + repr({
    key: value for key, value in bindings.items()
    if "Empirical" in str(key)
}), flush=True)
print("registry_parameter_binding=" + repr(local_bindings.get("registry")), flush=True)
print("registry_field_values=" + repr([
    (ast.dump(expression), annotated, None if scope is None else scope.name)
    for expression, annotated, scope
    in graph_express2._class_body_field_values(definition, "registry")
]), flush=True)
receiver = graph_express2._resolve_ast_parent_reference(
    ast.parse("self.registry").body[0].value,
    {**bindings, "self": definition},
)
print("receiver=" + repr(receiver), flush=True)
print("solve_records=" + repr([
    record for record in unresolved if record.get("name") == "solve"
]), flush=True)
print("solve_links=" + repr([
    link for link in links if "solve" in repr(link)
]), flush=True)
