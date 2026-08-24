import _pickle

from src.compiler.shell_external_references import (
    EXTERNAL_CALL,
    EXTERNAL_OK,
    EXTERNAL_RELEASE,
    EXTERNAL_RESOLVE,
    ExistingModuleExternalReferenceHost,
    PythonShellExternalReferenceResolver,
    ExternalReferenceRequestRecord,
)


def test_python_shell_resolver_uses_record_abi_for_pickle():
    import pickle

    resolver = PythonShellExternalReferenceResolver()

    result = resolver.call(
        "_pickle.loads",
        (pickle.dumps(("compiled", 41)),),
        result_dtype="opaque_ref",
    )

    assert result == ("compiled", 41)
    assert resolver._reference_ids == {"_pickle.loads": 1}


def _request(host, memory, operation, payload, *, reference_id=0, request_id=1):
    argument_offset = 64
    result_offset = 1024
    memory[argument_offset:argument_offset + len(payload)] = payload
    request = ExternalReferenceRequestRecord(
        operation=operation,
        request_id=request_id,
        reference_id=reference_id,
        arguments_offset=argument_offset,
        arguments_length=len(payload),
        result_offset=result_offset,
        result_capacity=len(memory) - result_offset,
    )
    completion = host.service(request, memory)
    result = bytes(memory[result_offset:result_offset + completion.result_length])
    return request, completion, result


def test_existing_module_pickle_round_trips_through_external_reference_records():
    host = ExistingModuleExternalReferenceHost()
    memory = bytearray(4096)
    request, completion, result = _request(
        host, memory, EXTERNAL_RESOLVE, host.values.encode("_pickle.loads")
    )
    assert len(request.pack()) == 32
    assert len(completion.pack()) == 32
    assert completion.status == EXTERNAL_OK
    reference_id = host.values.decode(result)

    expected = {"native-boundary": [1, 2, 3]}
    _request_record, completion, result = _request(
        host,
        memory,
        EXTERNAL_CALL,
        host.values.encode_arguments((_pickle.dumps(expected),)),
        reference_id=reference_id,
        request_id=2,
    )
    assert completion.status == EXTERNAL_OK
    assert host.values.decode(result) == expected

    _request_record, completion, _result = _request(
        host, memory, EXTERNAL_RELEASE, b"",
        reference_id=reference_id, request_id=3,
    )
    assert completion.status == EXTERNAL_OK


def test_external_scalar_result_is_inline_not_an_object_handle():
    host = ExistingModuleExternalReferenceHost()
    memory = bytearray(4096)
    _request_record, completion, result = _request(
        host, memory, EXTERNAL_RESOLVE, host.values.encode("time.perf_counter")
    )
    reference_id = host.values.decode(result)
    _request_record, completion, result = _request(
        host,
        memory,
        EXTERNAL_CALL,
        host.values.encode_arguments(()),
        reference_id=reference_id,
        request_id=2,
    )
    assert completion.status == EXTERNAL_OK
    assert isinstance(host.values.decode(result), float)
