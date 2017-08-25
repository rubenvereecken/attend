# Condor variables can overwritten at run-time using 'var = value'
# e.g. condor_submit <( submit.py ) 'N = 5'

N = {{ N }}
max_retries = {{ max_retries }}

Executable  = {{ python }}
Universe = vanilla
Arguments = -m attend.train {{ args }}

{% if prefer == 'gpu' -%}
# Executable  = {{ python }}
# Arguments = -m attend.train {{ args }}
# Preference set to GPU
Requirements = Memory >= 4000 && CUDACapability >= 3.5
# Rank = CUDACapability
Rank = CUDAGlobalMemoryMb
{%- else -%}
# Temporary fix until I know how to force GPU off
# Executable = /vol/bitbucket/rv1017/attend/attend/condor/cpu_wrapper.sh
# Arguments = {{ python }} -m attend.train {{ args }}
# Preference set to CPU
Requirements = Memory >= 4000
Rank = KFlops
request_gpus = 0
{%- endif %}

Error = condor_logs/{{ prefix }}$(cluster).err
Output = condor_logs/{{ prefix }}$(cluster).out
Log = condor_logs/{{ prefix }}$(cluster).log

initialdir = {{ base }}

should_transfer_files = YES
# TODO all the time plis
when_to_transfer_output = ON_EXIT
max_retries = $(max_retries)

getenv = True

#environment = "{% for k, v in env.items() -%}{{k}}={{v}} {%- endfor %}"
environment = "{{env_string}}"

Requirements = (Arch == "INTEL" && OpSys == "LINUX") || \
               (Arch == "X86_64" && OpSys =="LINUX")

Queue $(N)
