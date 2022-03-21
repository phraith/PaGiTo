import "./App.css";
import MiniDrawer from "./components/Drawer/MiniDrawer";
import { Box, CssBaseline, Grid, MenuItem, Select } from "@mui/material";
import ScatterImage from "./components/ScatterImage/ScatterImage";
import GisaxsShapes from "./components/GisaxsShapes/GisaxsShapes";
import Instrumentation from "./components/Instrumentation/Instrumentation";
import UnitcellMeta from "./components/UnitcellMeta/UnitcellMeta";
import React, { useEffect, useState } from "react";
import Sample from "./components/Sample/Sample";

import {
  HttpTransportType,
  HubConnection,
  HubConnectionBuilder,
  HubConnectionState,
  LogLevel,
} from "@microsoft/signalr";

const App = () => {
  const colors = [
    "twilightShifted",
    "twilight",
    "autumn",
    "parula",
    "bone",
    "cividis",
    "cool",
    "hot",
    "hsv",
    "inferno",
    "jet",
    "magma",
    "ocean",
    "pink",
    "plasma",
    "rainbow",
    "spring",
    "summer",
    "viridis",
    "winter",
  ];

  const [connection, _] = useState<HubConnection>(new HubConnectionBuilder()
  .withUrl("/message", {
    skipNegotiation: true,
    transport: HttpTransportType.WebSockets,
    accessTokenFactory: () => {
      return `${localStorage.getItem("apiToken")}`;
    },
  })
  .configureLogging(LogLevel.Information)
  .withAutomaticReconnect()
  .build());

  const [intensities, setIntensities] = useState<string>();

  useEffect(() => {
    const receiveJobResult = (message: any) => {
      let url = "/api/redis?" + message;
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

    if (connection) {
        connection.start()
            .then(result => {
                console.log('Connected!');

                connection.on('ReceiveJobId', message => {
                  receiveJobResult(message);
                });
            })
            .catch(e => console.log('Connection failed: ', e));
    }
}, [connection]);



  const handleData = (input: any) => {
    var startTime = performance.now();
    let json = JSON.parse(input);
    setIntensities(json.data);
    var endTime = performance.now();
    console.log(`Handling data took ${endTime - startTime} milliseconds`);
  };

  const [colormap, setColorMap] = React.useState("twilightShifted");
  const [jsonData, setJsonData] = React.useState({});

  const handleColorChange = (event) => {
    setColorMap(event.target.value as string);
  };

  const jsonCallback = (value, key) => {
    jsonData[key] = value;
    setJsonData({ ...jsonData });
  };

  useEffect(() => {
    if (connection?.state === HubConnectionState.Connected) {
      connection?.send(
        "IssueJob",
        JSON.stringify({
          info: {
            clientId: 0,
            jobId: 0,
            colormapName: colormap,
          },
          config: {
            ...jsonData,
          },
        })
      );
      console.log("after job sent");
    }
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
            }}
          >
            <ScatterImage intensities={intensities} />
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
                      <Select value={colormap} onChange={handleColorChange}>
                        {colors.map((value) => (
                          <MenuItem key={value} value={value}>
                            {value}
                          </MenuItem>
                        ))}
                      </Select>
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
    </React.Fragment>
  );
};

export default App;
