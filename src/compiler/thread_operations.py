"""Exact thread lifecycle vocabulary shared by ingestion and SSA."""

THREAD_OPERATORS = {
    "thread_create": "ThreadCreate",
    "thread_start": "ThreadStart",
    "thread_join": "ThreadJoin",
    "thread_is_alive": "ThreadIsAlive",
}

THREAD_RESULT_DTYPES = {
    "ThreadCreate": "opaque_ref",
    "ThreadStart": "none",
    "ThreadJoin": "none",
    "ThreadIsAlive": "bool",
}

PROCESS_OPERATORS = {
    **THREAD_OPERATORS,
    "barrier_create": "BarrierCreate", "barrier_wait": "BarrierWait",
    "barrier_abort": "BarrierAbort", "barrier_reset": "BarrierReset",
    "condition_create": "ConditionCreate", "condition_acquire": "ConditionAcquire",
    "condition_release": "ConditionRelease", "condition_wait": "ConditionWait",
    "condition_notify": "ConditionNotify", "condition_notify_all": "ConditionNotifyAll",
    "event_create": "EventCreate", "event_set": "EventSet", "event_clear": "EventClear",
    "event_wait": "EventWait", "event_is_set": "EventIsSet",
    "byte_queue_create": "ByteQueueCreate",
    "exception_none": "ExceptionNone", "exception_make": "ExceptionMake",
    "exception_take": "ExceptionTake", "exception_active": "ExceptionActive",
    "exception_match": "ExceptionMatch", "exception_raise": "ExceptionRaise",
    "exception_index_type": "ExceptionIndexType", "exception_length": "ExceptionLength",
}
PROCESS_RESULT_DTYPES = {
    **THREAD_RESULT_DTYPES,
    "BarrierCreate": "opaque_ref", "BarrierWait": "int64",
    "BarrierAbort": "none", "BarrierReset": "none",
    "ConditionCreate": "opaque_ref", "ConditionAcquire": "bool", "ConditionRelease": "none",
    "ConditionWait": "bool", "ConditionNotify": "none", "ConditionNotifyAll": "none",
    "EventCreate": "opaque_ref", "EventSet": "none", "EventClear": "none",
    "EventWait": "bool", "EventIsSet": "bool", "ByteQueueCreate": "opaque_ref",
    "ExceptionNone": "opaque_ref", "ExceptionMake": "opaque_ref", "ExceptionTake": "opaque_ref",
    "ExceptionActive": "bool", "ExceptionMatch": "bool", "ExceptionRaise": "none",
    "ExceptionIndexType": "bool", "ExceptionLength": "int64",
}
