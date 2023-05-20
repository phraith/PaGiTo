import Box from '@mui/material/Box/Box'
import Grid from '@mui/material/Grid/Grid'
import Sample from './Sample/Sample'
import GisaxsShapes from './GisaxsShapes/GisaxsShapes'
import ColormapSelect from './Colormap'
import UnitcellMeta from './UnitcellMeta/UnitcellMeta'
import Instrumentation from './Instrumentation/Instrumentation'

type Props = {
  colormap: any
  setColorMap: any
  jsonCallback: any
  isSimulation: boolean
}

const Settings = (props: Props) => {
  return (
    <Grid container direction={"row"} sx={{ height: "100%" }}>
      <Grid item xs={12} sm={12} md={12} lg={12} sx={{ height: "21vh" }}>
        <Box display="flex" flexDirection={"row"} sx={{ gap: 2, height: "100%" }}>
          <Box sx={{ width: "60%", height: "100%" }}>
            <Instrumentation jsonCallback={props.jsonCallback} />
          </Box>
          <Box display="flex" gap={2} flexDirection={"column"} sx={{ width: "40%", height: "100%" }}>
            <Box sx={{ height: "100%" }}>
              <UnitcellMeta jsonCallback={props.jsonCallback} />
            </Box>
            <ColormapSelect colormap={props.colormap} setColormap={props.setColorMap} />
          </Box>
        </Box>
      </Grid>
      <Grid item xs={12} sm={12} md={12} lg={12} paddingTop={2} sx={{ height: "calc(100% - 21vh)" }}>
        <Box display="flex" flexDirection={"row"} sx={{ gap: 2, height: "100%" }}>
          <Box sx={{ width: "60%", height: "100%" }}>
            <GisaxsShapes isSimulation={props.isSimulation} jsonCallback={props.jsonCallback} />
          </Box>
          <Box sx={{ width: "40%", height: "100%" }}>
            <Sample jsonCallback={props.jsonCallback} />
          </Box>
        </Box>
      </Grid>
    </Grid>
  )
}

export default Settings