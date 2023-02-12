import { DataGrid } from '@mui/x-data-grid/DataGrid';
import { Box, Button, CircularProgress, makeStyles, Typography } from '@mui/material';
import { memo, useState, useEffect } from 'react';
import { JobInfo } from '../../utility/JobInfo';
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutline';
import  { CircularProgressProps } from '@mui/material/CircularProgress';
interface JobsTableProps {
  setJob: any
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

  const renderConfigsButton = (params) => {
    return (
      <strong>
        <Button
          variant="contained"
          color="primary"
          size="small"
          onClick={() => props.setJob(new JobInfo(params.row.id, params.row.config))}
        >
          Show
        </Button>
      </strong>
    )
  }
  const renderResultButton = (params) => {

    return (
      <strong>{
        <Button
          variant="contained"
          color="primary"
          size="small"
          disabled={true}
          onClick={() => props.setJob(new JobInfo(params.row.id, params.row.config))}
        >
          Show
        </Button>}
      </strong>
    )
  }

  const renderProgressButton = (params) => {


    return (
      // <CheckCircleOutlineIcon color="primary" />
      <CircularProgressWithLabel value={20} />
    )
  }


  const columns = [
    { field: 'id', headerName: 'Job-ID', width: 300 },
    { field: 'config', headerName: "Config", renderCell: renderConfigsButton },
    { field: 'progress', headerName: "Progress", renderCell: renderProgressButton },
    { field: 'result', headerName: "Result", renderCell: renderResultButton }
  ]

  useEffect(() => {
    fetch("/api/jobs", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
      },
    })
      .then((data) => data.json())
      .then((data) => data.jobs.map(entry => {
        console.log(JSON.parse(entry.config))
        return {
          id: entry.jobId,
          config: JSON.parse(entry.config),
        }
      })
      )
      .then((data) => setTableData(data))
  }, [])


  return (
    <DataGrid sx={{
      height: 500, "&.MuiDataGrid-root .MuiDataGrid-cell:focus-within": {
        outline: "none !important",
      },
    }} rows={tableData}
      disableColumnSelector={true}

      columns={columns}
    />

  );
}

export default memo(JobsTable)