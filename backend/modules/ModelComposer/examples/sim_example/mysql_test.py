from sqlalchemy import *
from sqlalchemy.dialects.mysql import LONGBLOB
from sqlalchemy.orm import sessionmaker, declarative_base
import pandas as pd

Base = declarative_base()

class GisaxsImage(Base):
     __tablename__ = 'gisaxs_images'

     id = Column(Integer, primary_key=True)
     blob = Column(LONGBLOB)

     def __repr__(self):
            return f"User(id={self.id!r}, name={self.blob!r})"

def convertToBinaryData(filename):
    with open(filename, 'rb') as file:
        binaryData = file.read()
    return binaryData

# specify database configurations
config = {
    'host': '127.0.0.1',
    'port': 3306,
    'user': 'root',
    'password': 'pwd',
    'database': 'mysql'
}
db_user = config.get('user')
db_pwd = config.get('password')
db_host = config.get('host')
db_port = config.get('port')
db_name = config.get('database')# specify connection string
connection_str = f'mysql+pymysql://{db_user}:{db_pwd}@{db_host}:{db_port}/{db_name}'# connect to database
engine = create_engine(connection_str)
connection = engine.connect()# pull metadata of a table
metadata = MetaData(bind=engine)

Table(
   'gisaxs_images', metadata, 
   Column('id', Integer, primary_key = True), 
   Column('blob', LargeBinary),
)
metadata.create_all(engine)

# create a configured "Session" class
Session = sessionmaker(bind=engine)

# create a Session
session = Session()

# work with sess

data = convertToBinaryData("C:\\Users\\Phil\\Documents\\GISAXS-SimFit\\docker\\source\\modules\\ModelComposer\\examples\\sim_example\\sim_cylinder_10nm_0sd\\sim_cylinder_10nm_0sd.jpeg")
print(len(data))
myobject = GisaxsImage(blob=data)
session.add(myobject)
session.commit()


user_table = pd.read_sql_table(table_name="students", con=engine)
print(user_table)
