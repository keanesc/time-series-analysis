FROM ghcr.io/prefix-dev/pixi:latest

WORKDIR /app
COPY . .
RUN pixi install --locked && rm -rf ~/.cache/rattler

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8000/health').getcode()==200 else 1)"
CMD ["pixi", "run", "start"]
