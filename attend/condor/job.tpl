
Executable  = {{ python }}
Universe = vanilla
Arguments = -m attend.train {{ args }}

{% if prefer == 'gpu' -%}
# Preference set to GPU
Requirements = Memory >= 4000 && CUDACapability >= 3.5
# Rank = CUDACapability
Rank = CUDAGlobalMemoryMb
{%- else -%}
# Preference set to CPU
Requirements = Memory >= 4000
Rank = Mips
request_gpus = 0
{%- endif %}

Error = condor_logs/{{ prefix }}$(cluster).err
Output = condor_logs/{{ prefix }}$(cluster).out
Log = condor_logs/{{ prefix }}$(cluster).log

initialdir = {{ base }}

should_transfer_files = YES
# TODO all the time plis
when_to_transfer_output = ON_EXIT

getenv = True

#environment = "{% for k, v in env.items() -%}{{k}}={{v}} {%- endfor %}"
environment = "{{env_string}}"

Requirements = (Arch == "INTEL" && OpSys == "LINUX") || \
               (Arch == "X86_64" && OpSys =="LINUX")

Queue
