from app.services.tasks import celery_app, ping


def test_celery_app_and_ping_task_are_available() -> None:
    assert celery_app is not None
    assert ping.name == "app.services.tasks.ping"
