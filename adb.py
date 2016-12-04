import os
import subprocess

class ADB:
    def __init__(self):
        self.adbProcess = None
    def Shell(self, cmd):
        """Sends a single command by invoking a one-off command.
        Note you should use self.ShellInteractive for commands that
        need to be sent in rapid succession.
        """
        cmds = ['adb', 'shell', cmd]
        return subprocess.check_output(cmds)
    def OpenShell(self):
        """Opens an interactive shell and keeps it the pipe open for
        interactive commands."""
        cmds = ['adb', 'shell']
        p = subprocess.Popen(cmds, stdin=subprocess.PIPE, \
            stdout=open(os.devnull, 'wb'), stderr=subprocess.PIPE)
        self.adbProcess = p
    def ShellInteractive(self, cmd):
        """Pipes command to already open interactive shell. This
        should be used for time-sensitive commands."""
        if self.adbProcess == None:
            raise Exception("Shell wasn't opened first!")
        #print cmd
        self.adbProcess.stdin.write(cmd + '\n')
        self.adbProcess.stdin.flush()
    def CloseShell(self):
        self.adbProcess.kill()

