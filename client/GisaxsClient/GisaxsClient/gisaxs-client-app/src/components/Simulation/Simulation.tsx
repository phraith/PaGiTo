import MiniDrawer from "../Drawer/MiniDrawer";
import MenuItem from "@mui/material/MenuItem"
import FormControl from "@mui/material/FormControl"
import Grid from "@mui/material/Grid"
import Select from "@mui/material/Select"
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import ScatterImage from "../ScatterImage/ScatterImage";
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import React, * as react from "react";
import Sample from "../Sample/Sample";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import { useEffect, useState } from "react";
import ColormapSelect from "../Colormap";

const Simulation = () => {
  const receiveJobResult = (message: any) => {
    let url = "/api/redis/data?" + message;
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
      receiveJobResult,
      (message: string) => { },
      (message: string) => { }
    )
  )

  const [intensities, setIntensities] = react.useState<string>();
  const [imgWidth, setImgWidth] = react.useState<number>();
  const [imgHeight, setImgHeight] = react.useState<number>();

  useEffect(() => {
    hubConnection.connect()
  }, [hubConnection]);

  const handleData = (input: any) => {
    var startTime = performance.now();
    let json = JSON.parse(input);
    setIntensities(json.data);
    setImgWidth(json.width);
    setImgHeight(json.height);
    var endTime = performance.now();
    console.log(`Handling data took ${endTime - startTime} milliseconds`);
  };

  const [colormap, setColorMap] = React.useState("twilightShifted");
  const [jsonData, setJsonData] = React.useState({});

  const jsonCallback = (value, key) => {
    jsonData[key] = value;
    setJsonData({ ...jsonData });
  };

  react.useEffect(() => {
    let jsonConfig = JSON.stringify({
      info: {
        clientId: 0,
        jobId: 0,
        jobType: "sim",
        colormapName: colormap,
      },
      config: {
        ...jsonData,
      },
    });

    localStorage.setItem("simulation_config", jsonConfig);
    hubConnection.requestJob(jsonConfig);
  }, [jsonData, colormap]);

  return (
    <React.Fragment>
      <CssBaseline />
      <MiniDrawer />
      <Grid container spacing={2}>
        <Grid item xs={12} sm={12} md={12} lg={8}>
          <Box
            sx={{
              paddingTop: 10,
              paddingRight: 5,
              paddingLeft: 10,
              paddingBottom: 10,
            }}>
            <ScatterImage intensities={intensities} width={imgWidth} height={imgHeight} />
          </Box>
        </Grid>

        <Grid item xs={12} sm={12} md={12} lg={4}>
          <Grid
            container
            sx={{
              position: "sticky",
              top: 0,
              paddingTop: 10,
              paddingRight: 5,
              paddingLeft: 10,
            }}
          >
            <Grid item xs={12} sm={12} md={12} lg={12}>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={7} md={7} lg={7}>
                  <Instrumentation jsonCallback={jsonCallback} />
                </Grid>
                <Grid item xs={12} sm={5} md={5} lg={5}>
                  <Grid container rowSpacing={2}>
                    <Grid item xs={12}>
                      <UnitcellMeta jsonCallback={jsonCallback} />
                    </Grid>
                    <Grid item xs={12}>
                      <ColormapSelect colormap={colormap} setColormap={setColorMap} />
                    </Grid>
                  </Grid>
                </Grid>

                <Grid item xs={12} sm={7} md={7} lg={7}>
                  <GisaxsShapes jsonCallback={jsonCallback} />
                </Grid>
                <Grid item xs={12} sm={5} md={5} lg={5}>
                  <Sample jsonCallback={jsonCallback} />
                </Grid>
              </Grid>
            </Grid>
          </Grid>
        </Grid>
      </Grid>
    </React.Fragment >
  );
};

export default Simulation