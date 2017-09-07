# Condor variables can overwritten at run-time using 'var = value'
# e.g. condor_submit <( submit.py ) 'N = 5'

N = {{ N }}
max_retries = {{ max_retries }}

Executable  = {{ python }}
Universe = vanilla
Arguments = -m attend.train {{ args }}
{% if prefix and prefix != '' %}
+Prefix = {{ prefix }};
{% endif %}

{% if prefer == 'gpu' -%}
# Preference set to GPU
# Rank = CUDACapability
Rank = CUDAGlobalMemoryMb
req = CUDACapability >= 3.5
{%- else -%}
# Temporary fix until I know how to force GPU off
# Executable = /vol/bitbucket/rv1017/attend/attend/condor/cpu_wrapper.sh
# Arguments = {{ python }} -m attend.train {{ args }}
# Preference set to CPU
Rank = KFlops
request_gpus = 0
{%- endif %}
Requirements = ((Arch == "INTEL" && OpSys == "LINUX") || \
               (Arch == "X86_64" && OpSys =="LINUX")) && $(req)
request_memory = 4000

{%- if prefix and prefix != '' -%}
  {%- set prefix = prefix + '_' -%}
{%- else -%}
  {%- set prefix = '' -%}
{%- endif %}

Error = condor_logs/{{ prefix }}$(cluster).err
Output = condor_logs/{{ prefix }}$(cluster).out
Log = condor_logs/{{ prefix }}$(cluster).log

initialdir = {{ base }}

should_transfer_files = YES
# TODO all the time plis
when_to_transfer_output = ON_EXIT

next_job_start_delay = 1
max_retries = $(max_retries)
# Restart the job when exit code wasn't 0 and it wasn't killed
on_exit_remove = (ExitBySignal == False) && (ExitCode == 0)

getenv = True
environment = "{{env_string}}"

Queue $(N)
