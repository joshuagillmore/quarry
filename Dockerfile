# Pinned by tag, not digest: python:3.12-slim@sha256:... would be more
# reproducible (a tag can move), but constraints.txt is what actually freezes
# the dependency tree here, and a moving base-image tag still gets security
# patches. Revisit if supply-chain requirements tighten.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    gosu \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 -s /bin/bash app

WORKDIR /app

# requirements.txt names the direct dependencies; constraints.txt pins the
# entire tree (pip freeze of a verified-working image) so rebuilds are
# reproducible and an upstream release can't land silently — bleach is the
# app's XSS control, so surprise bumps there are security-relevant.
# Refresh deliberately: rebuild, verify, then `pip freeze > constraints.txt`.
COPY requirements.txt constraints.txt ./
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Fail the build here, not at first request, if the pinned litellm fork or
# crawl4ai can't actually be imported (e.g. a transitive dep mismatch).
RUN python -c "import litellm, crawl4ai"

# Install Chromium's OS-level dependencies as root. The unprivileged app user
# cannot apt-install, which is why crawl4ai-setup's dependency step failed
# (su: Authentication failure) and left Chromium unable to launch at runtime.
RUN python -m playwright install-deps chromium

RUN chown -R app:app /app
USER app

# Download the Chromium browser binary into /home/app/.cache/ms-playwright as the
# app user (OS deps already present from the root step). Do this explicitly —
# crawl4ai-setup aborts its own browser download when its dep-install step
# (which needs root) fails, leaving no Chromium binary at runtime.
RUN python -m playwright install chromium
# crawl4ai-setup's own dependency step needs root and used to fail silently
# (hence the old `|| true`); that step now runs as root above, so a failure
# here is real and should break the build. Follow it with an actual browser
# launch — a broken/missing Chromium binary must fail the build, not surface
# for the first time as a crawl error in production.
RUN crawl4ai-setup
RUN python -c "from playwright.sync_api import sync_playwright as p; b = p().start().chromium.launch(); b.close()"

USER root

COPY --chown=app:app . .
COPY --chmod=0755 entrypoint.sh /usr/local/bin/entrypoint.sh
# Only /app/data needs app ownership here (the bind mount target); everything
# else under /app was already chowned by the COPY --chown above, and
# re-chowning the whole tree on every build was needless I/O.
RUN mkdir -p /app/data && chown -R app:app /app/data

EXPOSE 5000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
# gunicorn.conf.py holds the server settings (workers=1 is load-bearing: see
# the comment there) and post_worker_init, which runs app.initialize() once
# per worker startup (init_db, reconcile_interrupted_missions, the scheduler)
# instead of racing the first inbound request through ensure_db's lock.
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
