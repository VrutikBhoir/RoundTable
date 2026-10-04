import asyncio

from backend.store import InMemorySessionStore


def test_started_session_accepts_late_join_request():
    async def scenario():
        store = InMemorySessionStore()
        credentials, record = store.create_session("Room", 3, "Host")
        await store.create_invitation(record)
        await store.start(record)

        request = await store.request_join(record, record.invitation.token, "Charlie")
        assert request.status.value == "pending"

        approved = await store.approve_request(record, request.id)
        assert approved.status.value == "approved"
        assert approved.participant_id in record.participants

    asyncio.run(scenario())


def test_locked_session_rejects_late_join_request():
    async def scenario():
        store = InMemorySessionStore()
        _, record = store.create_session("Room", 3, "Host")
        await store.create_invitation(record)
        await store.set_locked(record, True)

        try:
            await store.request_join(record, record.invitation.token, "Charlie")
        except ValueError as error:
            assert str(error) == "session_locked"
        else:
            raise AssertionError("locked session accepted a join request")

    asyncio.run(scenario())


def test_started_session_can_be_locked_and_unlocked():
    async def scenario():
        store = InMemorySessionStore()
        _, record = store.create_session("Room", 3, "Host")
        await store.create_invitation(record)
        await store.start(record)

        await store.set_locked(record, True)
        assert record.status.value == "LOCKED"
        try:
            await store.request_join(record, record.invitation.token, "Charlie")
        except ValueError as error:
            assert str(error) == "session_locked"
        else:
            raise AssertionError("locked started session accepted a join request")

        await store.set_locked(record, False)
        assert record.status.value == "STARTED"
        request = await store.request_join(record, record.invitation.token, "Charlie")
        assert request.status.value == "pending"

    asyncio.run(scenario())
