executable = condor_script.sh
getenv  = true
error   = exp.error
log     = exp.log
notification = always
transfer_executable = false
request_GPUs = 1
request_memory = 4*2048
queue
