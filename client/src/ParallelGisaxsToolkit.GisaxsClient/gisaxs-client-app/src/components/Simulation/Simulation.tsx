import MiniDrawer from "../Drawer/MiniDrawer";
import Grid from "@mui/material/Grid"
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import * as react from "react";
import * as React from "react";
import Sample from "../Sample/Sample";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import { useEffect, useState } from "react";
import ColormapSelect from "../Colormap";

const Simulation = () => {
  const [colormap, setColorMap] = React.useState("twilightShifted");
  const [jsonData, setJsonData] = React.useState({});


  const receiveJobResult = (hash: any) => {
    const url = `/api/job/${hash}`;
    console.log(hash)
    fetch(url, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
      },
    })

      .then((response) => response.json())
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

  const [intensities, setIntensities] = react.useState<string>();
  const [imgWidth, setImgWidth] = react.useState<number>();
  const [imgHeight, setImgHeight] = react.useState<number>();

  useEffect(() => {
    hubConnection.connect()
  }, [hubConnection]);

  const handleData = (input: any) => {
    console.log(input)
    var startTime = performance.now();
    let json = JSON.parse(input.response);
    console.log(json.jpegResults[0].data)
    setIntensities(json.jpegResults[0].data);
    setImgWidth(json.jpegResults[0].width);
    setImgHeight(json.jpegResults[0].height);
    var endTime = performance.now();
    console.log(`Handling data took ${endTime - startTime} milliseconds`);
  };

  const jsonCallback = (value, key) => {
    jsonData[key] = value;
    setJsonData({ ...jsonData });
  };

  react.useEffect(() => {
    let jsonConfig = JSON.stringify({
      meta: {
        type: "simulation",
        notification: "receiveJobResult",
        colormap: colormap
      },
      properties: {
        intensityFormat: "greyscale",
        simulationTargets: []
      },
      config: {
        ...jsonData,
      },
    });
    console.log(jsonConfig)
    localStorage.setItem("simulation_config", jsonConfig);

    const requestOptions = {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
        Accept: "application/json",
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(
        {
          "jsonConfig": jsonConfig
        }
      )
    };
    const url = "/api/job";
    fetch(url, requestOptions)
      .then(data => console.log(data));
  }, [jsonData, colormap]);

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