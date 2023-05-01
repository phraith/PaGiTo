import MiniDrawer from "../Drawer/MiniDrawer";
import Grid from "@mui/material/Grid"
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import * as React from "react";
import Sample from "../Sample/Sample";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import { useEffect, useState } from "react";
import ColormapSelect from "../Colormap";
import useJsonCallback from "../../hooks/useJsonCallback";
import useJobEffect from "../../hooks/useJobEffect";

const Simulation = () => {
  const [colormap, setColorMap] = React.useState("twilightShifted");
  const [json, jsonCallback] = useJsonCallback();
  const reponse = useJobEffect(json, colormap);

  const receiveJobResult = (hash: any) => {
    const requestOptions = {
      method: 'POST',
      headers: 
      {
          Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
          Accept: "application/json",
          'Content-Type': 'application/json'
      },
      body: JSON.stringify(
          {
            jobId: hash,
            includeResult: true
          }
      )
  };

  fetch("/api/job/state", requestOptions)
  .then((data) => data.json())
  .then((data) => handleData(data));
  };

  const [hubConnection, _] = useState<MessageHubConnectionProvider>(
    new MessageHubConnectionProvider(
      `${localStorage.getItem("apiToken")}`,
      [
        ["receiveJobResult", (message: string) => receiveJobResult(message)]
      ]
    )
  )

  const [intensities, setIntensities] = React.useState<string>();

  useEffect(() => {
    hubConnection.connect()
  }, [hubConnection]);

  const handleData = (input: any) => {
    let startTime = performance.now();
    setIntensities(JSON.parse(input.job.result).jpegResults[0].data);
    let endTime = performance.now();
    console.log(`Handling data took ${endTime - startTime} milliseconds`);
  };

  return (
    <React.Fragment>
      <CssBaseline />
      <MiniDrawer />
      <Grid container spacing={2} direction={"row"} padding={10}>
        <Grid item xs={8} sm={8} md={8} lg={8}>
          <Box sx={{ height: "100%", width: "100%", position: "relative" }}>
            <Box component="img" src={intensities} sx={{ width: "100%", height: "100%", position: "absolute" }} />
          </Box>
        </Grid>
        <Grid item xs={4} sm={4} md={4} lg={4}>
          <Box display="flex" flexDirection={"column"} sx={{ gap: 2 }}>

            <Box display="flex" sx={{ gap: 2 }}>
              <Instrumentation jsonCallback={jsonCallback} />
              <UnitcellMeta jsonCallback={jsonCallback} />
            </Box>

            <Box display="flex" sx={{ gap: 2 }}>
              <ColormapSelect colormap={colormap} setColormap={setColorMap} />
            </Box>

            <Grid container spacing={2} sx={{ height: "60vh" }}>
              <Grid item xs={7} sm={7} md={7} lg={7} sx={{ height: "100%" }}>
                <GisaxsShapes isSimulation={true} jsonCallback={jsonCallback} />
              </Grid>
              <Grid item xs={5} sm={5} md={5} lg={5} sx={{ height: "100%" }}>
                <Sample jsonCallback={jsonCallback} />
              </Grid>
            </Grid>
          </Box>
        </Grid>
      </Grid>
    </React.Fragment >
  );
};

export default Simulation