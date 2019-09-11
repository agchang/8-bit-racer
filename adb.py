import os
import subprocess


class ADB(object):
    def __init__(self):
        self.adb_process = None

    @staticmethod
    def shell(cmd):
        """Sends a single command by invoking a one-off command.
        Note you should use self.shell_interactive for commands that
        need to be sent in rapid succession.
        """
        cmds = ['adb', 'shell', cmd]
        return subprocess.check_output(cmds)

    def open_shell(self):
        """Opens an interactive shell and keeps it the pipe open for
        interactive commands."""
        cmds = ['adb', 'shell']
        proc = subprocess.Popen(cmds, stdin=subprocess.PIPE, \
            stdout=open(os.devnull, 'wb'), stderr=subprocess.PIPE)
        self.adb_process = proc

    def shell_interactive(self, cmd):
        """Pipes command to already open interactive shell. This
        should be used for time-sensitive commands."""
        if self.adb_process is None:
            raise Exception("shell wasn't opened first!")
        #print cmd
        self.adb_process.stdin.write(cmd.encode() + '\n'.encode())
        self.adb_process.stdin.flush()

    def close_shell(self):
        self.adb_process.kill()
