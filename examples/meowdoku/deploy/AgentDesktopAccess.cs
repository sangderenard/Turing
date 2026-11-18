using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Threading;

// CreateProcessWithLogonW inherits the caller's window station and desktop.
// In an OpenSSH Session 0, the agent account needs access to those objects
// before pwsh can initialize. Keep the ACEs only for the child process lifetime.
public sealed class AgentDesktopAccess : IDisposable
{
    private const uint DaclSecurityInformation = 4;
    private const int StationAccess = 0x000F006E;
    private const int DesktopAccess = 0x000F00CF;
    private readonly Mutex mutex;
    private readonly IntPtr station;
    private readonly IntPtr desktop;
    private readonly SecurityIdentifier sid;
    private bool stationAdded;
    private bool desktopAdded;
    private bool disposed;

    private AgentDesktopAccess(Mutex mutex, IntPtr station, IntPtr desktop, SecurityIdentifier sid)
    {
        this.mutex = mutex;
        this.station = station;
        this.desktop = desktop;
        this.sid = sid;
    }

    public static AgentDesktopAccess Grant(string account)
    {
        var sid = (SecurityIdentifier)new NTAccount(Environment.MachineName, account)
            .Translate(typeof(SecurityIdentifier));
        var mutex = new Mutex(false, @"Local\SpeakToMeAgentDesktopAccess");
        bool ownsMutex = false;
        try
        {
            try
            {
                if (!mutex.WaitOne(TimeSpan.FromMinutes(5)))
                    throw new TimeoutException("Timed out waiting for agent desktop access");
            }
            catch (AbandonedMutexException) { /* The abandoned mutex is acquired. */ }
            ownsMutex = true;

            IntPtr station = GetProcessWindowStation();
            IntPtr desktop = GetThreadDesktop(GetCurrentThreadId());
            if (station == IntPtr.Zero || desktop == IntPtr.Zero)
                throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot find the current window station or desktop");

            var lease = new AgentDesktopAccess(mutex, station, desktop, sid);
            try
            {
                lease.stationAdded = AddAce(station, sid, StationAccess);
                lease.desktopAdded = AddAce(desktop, sid, DesktopAccess);
                return lease;
            }
            catch
            {
                try { lease.Dispose(); }
                finally { ownsMutex = false; }
                throw;
            }
        }
        catch
        {
            if (ownsMutex) mutex.ReleaseMutex();
            mutex.Dispose();
            throw;
        }
    }

    public void Dispose()
    {
        if (disposed) return;
        disposed = true;
        try
        {
            if (desktopAdded) RemoveAce(desktop, sid, DesktopAccess);
        }
        finally
        {
            try { if (stationAdded) RemoveAce(station, sid, StationAccess); }
            finally
            {
                mutex.ReleaseMutex();
                mutex.Dispose();
            }
        }
    }

    private static RawSecurityDescriptor Read(IntPtr handle)
    {
        uint securityInfo = DaclSecurityInformation;
        uint needed;
        GetUserObjectSecurity(handle, ref securityInfo, null, 0, out needed);
        if (needed == 0)
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot size user object security descriptor");
        var data = new byte[needed];
        if (!GetUserObjectSecurity(handle, ref securityInfo, data, (uint)data.Length, out needed))
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot read user object security descriptor");
        return new RawSecurityDescriptor(data, 0);
    }

    private static void Write(IntPtr handle, RawSecurityDescriptor descriptor)
    {
        var data = new byte[descriptor.BinaryLength];
        descriptor.GetBinaryForm(data, 0);
        uint securityInfo = DaclSecurityInformation;
        if (!SetUserObjectSecurity(handle, ref securityInfo, data))
            throw new Win32Exception(Marshal.GetLastWin32Error(), "Cannot update user object security descriptor");
    }

    private static bool Matches(GenericAce ace, SecurityIdentifier sid, int mask)
    {
        var allowed = ace as CommonAce;
        return allowed != null && allowed.AceQualifier == AceQualifier.AccessAllowed &&
            allowed.SecurityIdentifier.Equals(sid) && allowed.AccessMask == mask && allowed.AceFlags == AceFlags.None;
    }

    private static bool AddAce(IntPtr handle, SecurityIdentifier sid, int mask)
    {
        var descriptor = Read(handle);
        var acl = descriptor.DiscretionaryAcl;
        if (acl == null) return false; // A null DACL already grants access.
        foreach (GenericAce ace in acl)
            if (Matches(ace, sid, mask)) return false;
        var updated = new RawAcl(acl.Revision, acl.Count + 1);
        foreach (GenericAce ace in acl) updated.InsertAce(updated.Count, ace);
        updated.InsertAce(updated.Count,
            new CommonAce(AceFlags.None, AceQualifier.AccessAllowed, mask, sid, false, null));
        descriptor.DiscretionaryAcl = updated;
        Write(handle, descriptor);
        return true;
    }

    private static void RemoveAce(IntPtr handle, SecurityIdentifier sid, int mask)
    {
        var descriptor = Read(handle);
        var acl = descriptor.DiscretionaryAcl;
        if (acl == null) return;
        for (int i = acl.Count - 1; i >= 0; i--)
        {
            if (!Matches(acl[i], sid, mask)) continue;
            acl.RemoveAce(i);
            descriptor.DiscretionaryAcl = acl;
            Write(handle, descriptor);
            return;
        }
    }

    [DllImport("user32.dll", SetLastError = true)]
    private static extern IntPtr GetProcessWindowStation();

    [DllImport("user32.dll", SetLastError = true)]
    private static extern IntPtr GetThreadDesktop(uint threadId);

    [DllImport("kernel32.dll")]
    private static extern uint GetCurrentThreadId();

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool GetUserObjectSecurity(IntPtr handle, ref uint securityInfo,
        [Out] byte[] descriptor, uint length, out uint needed);

    [DllImport("user32.dll", SetLastError = true)]
    private static extern bool SetUserObjectSecurity(IntPtr handle, ref uint securityInfo,
        byte[] descriptor);
}
