namespace ParallelGisaxsToolkit.Gisaxs.Core.UserStore;

public interface IUserStore
{
    Task<IEnumerable<User>> Get();
    Task<IEnumerable<User>> Get(long id);
    Task Delete(long id);
    Task Insert(User user);
}