import sqlite3

conn = sqlite3.connect('usersdbtrial') 
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS users
          ([username] TEXT, [full_name] TEXT,
          [email] TEXT, [hashed_password] TEXT, [disabled] BOOL)
          ''')
        
c.execute('''
          INSERT INTO users
                VALUES
                ('trialuser',"Trial User",
                "info@trialuser.com","$2a$12$O/9CiF4Ul3WdEgDPCaYtt.r/QjA5kZORpZENxNkV4E8HuD/fZEnma",
                "False")
          ''')

conn.commit()

print("finished")
