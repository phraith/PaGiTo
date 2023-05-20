import { DataGrid } from '@mui/x-data-grid/DataGrid';
import { Box, Button, CircularProgress, Typography } from '@mui/material';
import { memo, useState, useEffect } from 'react';
import { CircularProgressProps } from '@mui/material/CircularProgress';
interface JobsTableProps {
  setJsonData: any
  resultNotifier: any
}

function CircularProgressWithLabel(
  props: CircularProgressProps & { value: number },
) {
  return (
    <Box sx={{ position: 'relative', display: 'inline-flex' }}>
      <CircularProgress variant="determinate" {...props} />
      <Box
        sx={{
          top: 0,
          left: 0,
          bottom: 0,
          right: 0,
          position: 'absolute',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}
      >
        <Typography
          variant="caption"
          component="div"
          color="text.secondary"
        >{`${Math.round(props.value)}%`}</Typography>
      </Box>
    </Box>
  );
}

const JobsTable = (props: JobsTableProps) => {

  const [tableData, setTableData] = useState([])
  const [tableCount, setTableCount] = useState(0)
  const [result, setResult] = useState({})

  const pageSize = 20;


  const renderConfigsButton = (params) => {
    return (
      <strong>
        <Button
          variant="contained"
          color="primary"
          size="small"
          onClick={() => fetchConfig(params.row.id)}
        >
          Show
        </Button>
      </strong>
    )
  }
  const renderResultButton = (params) => {

    return (
      <Button
        variant="contained"
        color="primary"
        size="small"
        disabled={!params.row.finishDate}
        onClick={() => fetchResult(params.row.id)}
      >
        Show
      </Button>
    )
  }

  const fetchResult = (jobId) => {
    const requestOptions1 = {
      method: 'POST',
      headers:
      {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(
        {
          jobId: jobId,
          includeResult: true
        }
      )
    };

    let url1 = "/api/job/state";
    fetch(url1, requestOptions1)
      .then((data) => data.json())
      .then((data) => {
        props.setJsonData(data.job.result)
      })
  }

  const fetchConfig = (jobId) => {
    const requestOptions1 = {
      method: 'POST',
      headers:
      {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(
        {
          jobId: jobId,
          includeConfig: true
        }
      )
    };

    let url1 = "/api/job/state";
    fetch(url1, requestOptions1)
      .then((data) => data.json())
      .then((data) => {
        props.setJsonData(data.job.config)
      })
  }


  const renderProgressButton = (params) => {
    return (
      <CircularProgressWithLabel value={20} />
    )
  }


  const columns = [
    { field: 'id', headerName: 'Job-ID', width: 100 },
    { field: 'issueDate', headerName: "IssueDate", width: 200 },
    { field: 'finishDate', headerName: "FinishDate", width: 200 },
    { field: 'config', headerName: "Config", renderCell: renderConfigsButton },
    // { field: 'progress', headerName: "Progress", renderCell: renderProgressButton },
    { field: 'result', headerName: "Result", renderCell: renderResultButton },

  ]

  useEffect(() => {
    fetch(`/api/jobs/count`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
      },
    })
      .then((data) => data.json())
      .then((json) => setTableCount(json.jobCount))
    updateTableData(0)
  }, [])

  useEffect(() => {

  }, [props.resultNotifier])


  const updateTableData = (page: any) => {
    fetch(`/api/jobs/count`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
      },
    })
      .then((data) => data.json())
      .then((json) => setTableCount(json.jobCount))

    console.log(page)
    fetch(`/api/jobs/${page}/${pageSize}`, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
      },
    })
      .then((data) => data.json())
      .then((data) => data.jobs.map(entry => {
        return {
          id: entry.jobId,
          issueDate: entry.issueDate,
          finishDate: entry.finishDate
        }
      })
      )
      .then((data) => { console.log(data); setTableData(data) })
  }

  return (
    <DataGrid sx={{
      height: 500, "&.MuiDataGrid-root .MuiDataGrid-cell:focus-within": {
        outline: "none !important",
      }
    }}
      initialState={{
        sorting: {
          sortModel: [{ field: 'issueDate', sort: 'desc' }],
        },
      }}
      rows={tableData}
      paginationMode="server"
      rowCount={tableCount}
      pageSize={pageSize}
      disableColumnSelector={true}
      columns={columns}
      onPageChange={(p) => updateTableData(p)}
    />
  );
}

export default memo(JobsTable)