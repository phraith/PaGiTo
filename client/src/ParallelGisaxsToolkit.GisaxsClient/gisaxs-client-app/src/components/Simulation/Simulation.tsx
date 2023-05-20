import MiniDrawer from "../Drawer/MiniDrawer";
import Grid from "@mui/material/Grid"
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import * as React from "react";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import { useEffect, useState } from "react";
import useJsonCallback from "../../hooks/useJsonCallback";
import useJobEffect from "../../hooks/useJobEffect";
import Settings from "../Settings";

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
      <Box sx={{ height: "100vh" }}>
        <Grid container spacing={2} direction={"row"} padding={10} sx={{ height: "100%" }}>
          <Grid item xs={8} sm={8} md={8} lg={8} sx={{ height: "100%" }}>
            <Box sx={{ height: "100%", width: "100%", position: "relative" }}>
              <Box component="img" src={intensities} sx={{ width: "100%", height: "100%", position: "absolute" }} />
            </Box>
          </Grid>
          <Grid item xs={4} sm={4} md={4} lg={4} sx={{ height: "100%" }}>
            <Box sx={{ height: "100%" }}>
              <Settings isSimulation={true} jsonCallback={jsonCallback} colormap={colormap} setColorMap={setColorMap} />
            </Box>
          </Grid>
        </Grid>
      </Box>
    </React.Fragment >
  );
};

export default Simulation