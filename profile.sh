# https://docs.ray.io/en/latest/ray-contribute/profiling.html
export PERFTOOLS_PATH=/usr/lib/x86_64-linux-gnu/libprofiler.so
export PERFTOOLS_LOGFILE=/tmp/pprof.out

RAY_JEMALLOC_CONF=prof:true,lg_prof_interval:33,lg_prof_sample:17,prof_final:true,prof_leak:true \
RAY_JEMALLOC_LIB_PATH=~/jemalloc-5.2.1/lib/libjemalloc.so \
RAY_JEMALLOC_PROFILE=gcs_server \

ray start --head
# Use the appropriate path.
RAYLET=ray/python/ray/core/src/ray/raylet/raylet

google-pprof -svg $RAYLET /tmp/pprof.out > /tmp/pprof.svg
# Then open the .svg file with Chrome.

cat /tmp/pprof.out
