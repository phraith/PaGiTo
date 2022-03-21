using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace UserDataProvider
{
    public class UserDataContext : DbContext
    {
        public DbSet<User> Users { get; set; }
    
        public UserDataContext(DbContextOptions<UserDataContext> options)
            :
            base(options)
        { 

        }
    }
}
