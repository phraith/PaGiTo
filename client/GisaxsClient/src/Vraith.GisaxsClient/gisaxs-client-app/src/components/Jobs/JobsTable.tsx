import { useState, useEffect } from 'react';
import { DataGrid } from '@mui/x-data-grid/DataGrid';
import { Box } from '@mui/material';
import React from 'react';
import { ImageInfo } from '../../utility/ImageInfo';
import { JobInfo } from '../../utility/JobInfo';

interface JobsTableProps {
  setJobsInfo: any
}

const JobsTable = (props: JobsTableProps) => {

  const [tableData, setTableData] = useState([])

  const columns = [
    { field: 'id', headerName: 'ID' }
  ]

  useEffect(() => {
    fetch("/api/jobstore/info", {
      method: "GET",
      headers: {
          Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
          Accept: "application/json",
      },
  })
      .then((data) => data.json())
      .then((data) => data.map(entry => {
        return {
          id: entry.id,
          info: entry.jobInfo,
        }
      })
      )
      .then((data) => setTableData(data))
  }, [])

  return (
      <DataGrid sx={{height: 500}} rows={tableData}
        columns={columns}
        onRowClick={(e) => { console.log(e); props.setJobsInfo(new JobInfo(e.row.id, e.row.info)) }}
      />

  );
}

export default React.memo(JobsTable)