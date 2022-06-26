import { useState, useEffect } from 'react';
import { DataGrid } from '@mui/x-data-grid/DataGrid';
import { Box } from '@mui/material';
import React from 'react';
import { ImageInfo } from '../../utility/ImageInfo';

interface ImageTableProps {
  setImageInfo: any
}

const ImageTable = (props: ImageTableProps) => {

  const [tableData, setTableData] = useState([])

  const columns = [
    { field: 'id', headerName: 'ID' },
    { field: 'width', headerName: 'Width' },
    { field: 'height', headerName: 'Height' },
    { field: 'name', headerName: 'Image name', width: 300 },
    { field: 'size', headerName: 'Size in bytes', width: 600 }
  ]

  useEffect(() => {
    fetch("/api/scatterstore/info")
      .then((data) => data.json())
      .then((data) => setTableData(data))
  }, [])

  return (
    <Box style={{
      display: 'flex', paddingTop: 10,
      paddingRight: 5,
      paddingLeft: 10,
      paddingBottom: 10,
      height: 500
    }}>
      <DataGrid rows={tableData}
        columns={columns}
        onRowClick= {(e) => { console.log(e); props.setImageInfo(new ImageInfo(e.row.id, e.row.width, e.row.height))}}
      />
    </Box>

  );
}

export default React.memo(ImageTable)