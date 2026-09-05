from src.compiler.ssa_fortran_backend import sin_table_declaration


def test_baked_sine_table_is_continued_within_free_form_line_width():
    declaration = sin_table_declaration()

    lines = declaration.splitlines()
    assert len(lines) > 2
    assert all(len(line) <= 120 for line in lines)
    assert lines[0].endswith("[ &")
    assert all(line.endswith(", &") for line in lines[1:-1])
    assert lines[-1].endswith(" ]")
