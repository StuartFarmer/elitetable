FROM python:3.14.7-slim
RUN apt-get update && apt-get install -y --no-install-recommends g++ swig && rm -rf /var/lib/apt/lists/*
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/matplotlib
COPY requirements-worker.txt /tmp/requirements-worker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-worker.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.0+cpu'
RUN mkdir -p /opt/worker/rsikit/sandbox && printf 'from .policy import Policy\n' > /opt/worker/rsikit/__init__.py
COPY rsikit/policy.py rsikit/episode.py /opt/worker/rsikit/
COPY rsikit/sandbox/ /opt/worker/rsikit/sandbox/
RUN python -c "import site; from pathlib import Path; Path(site.getsitepackages()[0], 'rsikit-worker.pth').write_text('/opt/worker\n')"
RUN python /opt/worker/rsikit/sandbox/check_libraries.py > /opt/worker/libraries.json
USER 65534:65534
ENTRYPOINT ["python", "-I", "-u", "-c", "from rsikit.sandbox.worker import main; main()"]
