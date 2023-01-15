import { DataGrid } from '@mui/x-data-grid/DataGrid';
import { Box } from '@mui/material';
import { memo, useState, useEffect } from 'react';
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
    fetch("/api/images", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
      },
    })
      .then((data) => data.json())
      .then((data) => data.imageInfosWithId.map(entry => {
        return {
          id: entry.id,
          width: entry.info.width,
          height: entry.info.height,
          name: entry.info.name,
          size: entry.info.size,
        }
      })
      )
      .then((data) => setTableData(data))
  }, [])

  return (
      <DataGrid sx={{height: 500}} rows={tableData}
        columns={columns}
        onRowClick={(e) => { console.log(e); props.setImageInfo(new ImageInfo(e.row.id, e.row.width, e.row.height)) }}
      />

  );
}

export default memo(ImageTable)