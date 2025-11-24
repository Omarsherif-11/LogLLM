datasets = {
        "android": {
            "data_dir": "/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/datasets/android",
            "log_name": "Android.log",
            "safe_phrases": [
                "0 errors", "no error", "0 fail", "no fail", 
                "without error", "success", "completed", " verified ",
                "return code 0", "status: 0", "result: 0",
                "test", "simulation", "debug", "test", "trace",
                "low memory" # Common warning, usually handled by OS
            ],

            "critical_keywords": [
            # 1. High Confidence Crashes (Keep these)
                "fatal exception", "force close", "force finishing", "anr in", 
                "am_crash", "am_anr", "process died", "win death", "sigkill", 
                "sigsegv", "segmentation fault", "dead object", "kernel panic", 
                "watchdog reset", "system_server crash",
                
                # 2. Functional Failures (ADDED BACK)
                "failed",              # "Connection failed", "Failed to open"
                "failure",             # "Camera failure"
                "error",               # "System Error", "IO Error"
                "exception",           # "NullPointerException", "IndexOutOfBounds"
                "unable to",           # "Unable to connect"
                "could not",           # "Could not save"
                "denied",              # "Permission denied"
                "rejected",
                "aborted",
                "timed out",
                "timeout",
                "invalid",             # "Invalid argument"
                "corrupt"
            ]
        },
        "mac": {
            "data_dir": "/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/datasets/mac",
            "log_name": "Mac.log",
            "safe_phrases": [
                "0 errors", "no error", "0 fail", "no fail", 
                "without error", "success", "completed", "verified",
                "return code 0", "status: 0", "result: 0",
                "test", "simulation", "notice", "test", "debug", "trace",
                "low memory"
            ],

            "critical_keywords": [
            # 1. System Death (High Severity)
                "kernel panic", "segmentation fault", "core dump", 
                "deadlock", "assertion failed", "uncaught exception",
                "process killed", "out of memory", "stack overflow",
                "integrity check failed", "filesystem corruption",
                "kernel trap", "bad_address", "protection fault",
                "bus error", "illegal instruction", "panic:",
                
                # 2. Functional Failures (Medium Severity)
                "failed to", "failure", "corrupt", "unable to", "critical", 
                "fatal", "denied", "aborted", "dropping", "invalid", 
                "offline", "assocfail", "authfail", "authentication failed",
                "connection refused", "timed out",
                
                # 3. Extended Logic/Network Errors (New)
                "network is down", 
                "no network route", 
                "operation not permitted", 
                "paramerr",             # Mac specific parameter error
                "null key",             # Null pointer/key logic error
                "missing class name",   # Objective-C runtime error
                "scripting error",      # AppleScript/Cocoa error
                "error returned",       # "Error returned from..."
                "signalled",            # "Signalled (Error)"
                "identity not set"      # Account identity missing
            ]
        },
        "windows": {
            "data_dir": "/pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/datasets/windows",
            "log_name": "Windows.log",
            "safe_phrases": [
                "0 errors", "no error", "0 fail", "no fail", "succeeded", "successfully", 
                "completed", "verified", "return code 0", "status: 0", "result: 0",
                "informational", "audit success", "0x0", "service starts", "ready", "clean up",
                "operation successful", "run successfully", "hresult 0x00000000", "hresult 0x00000001",
                "test", "debug", "trace"
            ],

            "critical_keywords": [
                # 1. Windows Generic Failures (Broadened)
                "error",               # "Error code:", "Internal Error"
                "fail",                # "Failed to...", "Failure"
                "failed",
                "failure",
                "warning",             # "Warning: Failed to upload..."
                "exception",
                "denied",              # "Access denied"
                "invalid",             # "Invalid argument"
                "unable to",
                "could not",
                "abort",
                
                # 2. Specific Codes & Components
                "hresult",             # Catch all HRESULT failures (often 0x8...)
                "0x800",               # Failure hex prefix
                "e_fail",
                "terminated unexpectedly",
                "hung",
                "stopped",             # "Service stopped"
                "fatal",
                " critical ",
                "corruption",
                "blue screen",
                "bugcheck",
                "cbs error",           # Component Based Servicing error
                "csi error"            # Component Servicing Infrastructure error
            ]
        }
    }