// Windows launcher stub, compiled by launcher/pack_windows.ps1 with the .NET Framework csc.exe
// that ships with Windows. The packer embeds three resources: the launcher runtime
// (dsv41_launcher.py), its baked config (launcher.json) and the Python path (python.txt).
// The stub writes them beside the .exe (one stable folder per launcher) and runs Python
// in this console window, which is the launcher's status window.
using System;
using System.Diagnostics;
using System.IO;
using System.Reflection;
using System.Text;

static class LauncherStub
{
    static string Extract(Assembly self, string name, string folder)
    {
        string target = Path.Combine(folder, name);
        using (Stream source = self.GetManifestResourceStream(name))
        using (FileStream output = File.Create(target))
        {
            source.CopyTo(output);
        }
        return target;
    }

    static int Main()
    {
        Assembly self = Assembly.GetExecutingAssembly();
        string exe = self.Location;
        string folder = Path.Combine(Path.GetDirectoryName(exe), Path.GetFileNameWithoutExtension(exe) + ".runtime");
        try
        {
            Directory.CreateDirectory(folder);
            string script = Extract(self, "dsv41_launcher.py", folder);
            string config = Extract(self, "launcher.json", folder);
            string python;
            using (StreamReader reader = new StreamReader(self.GetManifestResourceStream("python.txt"), Encoding.UTF8))
            {
                python = reader.ReadToEnd().Trim();
            }
            // Ctrl+C belongs to the Python launcher (graceful stop); the stub just waits for it.
            Console.CancelKeyPress += delegate(object sender, ConsoleCancelEventArgs e) { e.Cancel = true; };
            ProcessStartInfo start = new ProcessStartInfo(python, "-B \"" + script + "\" --config \"" + config + "\"");
            start.UseShellExecute = false;
            start.WorkingDirectory = folder;
            Process child = Process.Start(start);
            child.WaitForExit();
            return child.ExitCode;
        }
        catch (Exception error)
        {
            Console.WriteLine("Launcher failed to start: " + error.Message);
            Console.WriteLine("Press Enter to close.");
            Console.ReadLine();
            return 1;
        }
    }
}
