from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


with DAG('credit_ingest', start_date=datetime(2025,1,1), schedule_interval='@daily', catchup=False) as dag:
	def ingest_batch():
		# connect to source, move to lake
		pass


	task = PythonOperator(task_id='ingest_batch', python_callable=ingest_batch)