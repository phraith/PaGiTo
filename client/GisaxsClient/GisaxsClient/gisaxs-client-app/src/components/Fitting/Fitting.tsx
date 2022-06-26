import MiniDrawer from "../Drawer/MiniDrawer";
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import Grid from "@mui/material/Grid"
import Select from "@mui/material/Select"
import throttle from "lodash/throttle";

import ScatterImage from "../ScatterImage/ScatterImage";
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import React, { useEffect, useMemo, useRef, useState } from "react";
import Sample from "../Sample/Sample";
import {
    HttpTransportType,
    HubConnection,
    HubConnectionBuilder,
    HubConnectionState,
    LogLevel,
} from "@microsoft/signalr";
import LineProfileWrapper from "../ScatterImage/LineProfileWrapper";
import { Coordinate, LineProfile, LineProfileState, RelativeLineProfile } from "../../utility/LineProfile";
import ImageTable from "./ImageTable";
import MenuItem from "@mui/material/MenuItem";
import { Button, Menu } from "@mui/material";
import LineProfileGraphVx from "./LineProfileGraphVx";
import { ImageInfo } from "../../utility/ImageInfo";


const Fitting = () => {
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

    const [connection, _] = useState<HubConnection>(
        new HubConnectionBuilder()
            .withUrl("/message", {
                skipNegotiation: true,
                transport: HttpTransportType.WebSockets,
                accessTokenFactory: () => {
                    return `${localStorage.getItem("apiToken")}`;
                },
            })
            .configureLogging(LogLevel.Information)
            .withAutomaticReconnect()
            .build()
    );

    const [intensities, setIntensities] = useState<string>();
    const [refIntensities, setRefIntensities] = useState<string>();
    const [currentInfoPath, setCurrentInfoPath] = useState<string>();
    const [imgWidth, setImgWidth] = useState<number>();
    const [imgHeight, setImgHeight] = useState<number>();
    const [lineprofileState, setLineprofileState] = useState<LineProfileState>(new LineProfileState(false, [], new RelativeLineProfile(new Coordinate(0, 0), new Coordinate(0, 1), new Coordinate(0, 0))));
    const [plotData, setPlotData] = React.useState([])

    const [openTable, setOpenTable] = React.useState<boolean>(false)
    const [imageInfo, setImageInfo] = React.useState<ImageInfo>(new ImageInfo(0,0,0))


    useEffect(() => {
        if (connection) {
            connection
                .start()
                .then((result) => {
                    console.log("Connected!");

                    connection.on("ReceiveJobId", (message) => {
                        receiveJobResult(message);
                    });

                    connection.on("ReceiveJobInfos", (message) => {
                        receiveJobInfos(message);
                    });

                    connection.on("ProcessLineprofiles", (message) => {
                        getLineprofiles(message)
                    })
                })
                .catch((e) => console.log("Connection failed: ", e));
        }
    }, [connection]);

    const receiveJobInfos = (message: any) => {
        setCurrentInfoPath(message)
        setIsActive(true)
    }

    const getLineprofiles = (message: any) => {
        let j = JSON.parse(message)
        let traces = []
        j["profiles"].forEach((element, i) => {
            let values = element.Data
            let k = values.map((x: number, index: number) => { return { x: index, y: x } })
            traces.push(k)
        });
        setPlotData(traces[0])
    }

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

    const handleData = (input: any) => {
        var startTime = performance.now();
        let json = JSON.parse(input);
        setIntensities(json.data);
        setImgWidth(json.width);
        setImgHeight(Math.min(json.height, window.innerHeight));
        var endTime = performance.now();
        console.log(`Handling data took ${endTime - startTime} milliseconds`);
    };

    const [colormap, setColorMap] = React.useState("twilightShifted");
    const [jsonData, setJsonData] = React.useState({});
    const [isActive, setIsActive] = React.useState(false)



    const handleColorChange = (event) => {
        setColorMap(event.target.value as string);
    };

    const jsonCallback = (value, key) => {
        jsonData[key] = value;
        setJsonData({ ...jsonData });
    };


    const sendLineprofileRequest = (data, lineprofiles) => {
        let jsonConfig = JSON.stringify({
            profiles: {
                ...lineprofiles
            },
            config: {
                ...data,
            },
        });

        if (connection?.state === HubConnectionState.Connected) {
            connection?.send("GetProfiles", jsonConfig);
            console.log("after profiles sent");
        }
    }

    const throttled = useRef(throttle((data, lineprofiles) => sendLineprofileRequest(data, lineprofiles), 500));

    useEffect(() => {
        throttled.current(jsonData, [lineprofileState?.currentLineProfile])
    }, [lineprofileState?.currentLineProfile]);

    useEffect(() => {
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

        if (connection?.state === HubConnectionState.Connected) {
            connection?.send("IssueJob", jsonConfig);
            console.log("after job sent");
        }
    }, [jsonData, colormap]);

    useEffect(() => {
        let url = "/api/scatterstore/get?id=" + imageInfo.id;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => setRefIntensities(data));
    }, [imageInfo.id]);


    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2}>
                <Grid item xs={6} sm={6} md={6} lg={4}>
                    <Box
                        sx={{
                            paddingTop: 10,
                            paddingBottom: 10,
                            paddingLeft: 10
                        }}>
                        <LineProfileWrapper key={"test2"} width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={(state) => {
                            setLineprofileState(state)
                        }}>
                            <ScatterImage key={"test2"} intensities={intensities} width={imgWidth} height={imgHeight} />
                        </LineProfileWrapper>
                    </Box>
                </Grid>
                <Grid item xs={6} sm={6} md={6} lg={4}>
                    <Box
                        sx={{
                            paddingTop: 10,
                            paddingRight: 5,
                            paddingBottom: 10,
                            paddingLeft: 5
                        }}>
                        <LineProfileWrapper key={"test2"} width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={(state) => {
                            setLineprofileState(state)
                        }}>
                            <ScatterImage key={"test2"} intensities={refIntensities} width={imageInfo.width} height={imageInfo.height} />
                        </LineProfileWrapper>
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

                        {!openTable &&
                            <Grid item xs={12} sm={12} md={12} lg={12} >
                                <LineProfileGraphVx data={plotData} ></LineProfileGraphVx>
                            </Grid>
                        }

                        {openTable &&
                            <Grid item xs={12} sm={12} md={12} lg={12}>
                                <ImageTable setImageInfo={(updatedImageInfo: ImageInfo) => { console.log(updatedImageInfo); setImageInfo(updatedImageInfo)}} />
                            </Grid>
                        }

                        <Grid item xs={12} sm={12} md={12} lg={12}>
                            <Grid container spacing={2}>
                                <Grid item xs={12} sm={7} md={7} lg={7}>
                                    <Instrumentation jsonCallback={jsonCallback} initialResX={imageInfo.width} initialResY={imageInfo.height} />
                                </Grid>
                                <Grid item xs={12} sm={5} md={5} lg={5}>
                                    <Grid container rowSpacing={2}>
                                        <Grid item xs={12}>
                                            <UnitcellMeta jsonCallback={jsonCallback} />
                                        </Grid>
                                        <Grid item xs={6}>
                                            <Select value={colormap} onChange={handleColorChange}>
                                                {colors.map((value) => (
                                                    <MenuItem key={value} value={value}>
                                                        {value}
                                                    </MenuItem>
                                                ))}
                                            </Select>
                                        </Grid>
                                        <Grid item xs={12} sm={12} md={12} lg={6} sx={{ justifyContent: "flex-end", display: 'flex' }}>
                                            <Button onClick={() => { setOpenTable(prevState => !prevState) }}>
                                                {openTable ? "Show graph" : "Show images"}
                                            </Button>
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

export default Fitting