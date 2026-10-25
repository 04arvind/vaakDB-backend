
from logging.config import fileConfig

from alembic import context

from app.core.config import settings
from app.db.models import Base
from app.db.session import engine


config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Tell Alembic which SQLAlchemy models to inspect.
target_metadata = Base.metadata

# Read the database URL from the existing application settings.
# Do not hardcode database credentials in alembic.ini.
config.set_main_option(
    "sqlalchemy.url",
    settings.database_url.replace("%", "%%"),
)


def run_migrations_offline() -> None:
    """Generate migration SQL without opening a database connection."""
    context.configure(
        url=settings.database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations using the application's SQLAlchemy engine."""
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
