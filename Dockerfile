# Pinned environment for reproducing every table and figure in this paper.
# The search statistics (Section~sec:setup-metrics, Appendix~sec:repro) are
# sensitive to MuJoCo's minor version, so this pins the interpreter and every
# dependency exactly, rather than leaving them floating.
#
# Build:  docker build -t typed-case-role-grammars .
# Run:    docker run --rm -it -v $(pwd)/results_out:/work/results_out \
#             typed-case-role-grammars bash
# Then follow REPRODUCE.md from inside the container.

FROM python:3.13.16-slim

WORKDIR /work

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash"]
