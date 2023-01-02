import MiniDrawer from "../Drawer/MiniDrawer";
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import Grid from "@mui/material/Grid"
import throttle from "lodash/throttle";
import ScatterImage from "../ScatterImage/ScatterImage";
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import React, { useEffect, useRef, useState } from "react";
import Sample from "../Sample/Sample";
import LineProfileWrapper from "../ScatterImage/LineProfileWrapper";
import { Coordinate, LineProfileState, RelativeLineProfile } from "../../utility/LineProfile";
import ImageTable from "./ImageTable";
import { Button } from "@mui/material";
import LineProfileGraphVx from "./LineProfileGraphVx";
import { ImageInfo } from "../../utility/ImageInfo";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import ColormapSelect from "../Colormap";


const Fitting = () => {
    const receiveJobInfos = (message: any) => {
        setCurrentInfoPath(message)
        setIsActive(true)
    }

    const getLineprofiles = (hash: any) => {
        let url = `/api/redis/data?hash=${hash}`;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => {
                let json = JSON.parse(data);
                let traces = []
                let values = json.modifiedData
                let k = values.map((x: number, index: number) => { return { x: index, y: x } })
                traces.push(k)
                setSimulatedPlotData(traces[0])
            })
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

    const [hubConnection, _] = useState<MessageHubConnectionProvider>(
        new MessageHubConnectionProvider(
            `${localStorage.getItem("apiToken")}`,
            receiveJobResult,
            receiveJobInfos,
            getLineprofiles
        )
    )

    const [intensities, setIntensities] = useState<string>();
    const [refIntensities, setRefIntensities] = useState<string>();
    const [currentInfoPath, setCurrentInfoPath] = useState<string>();
    const [imgWidth, setImgWidth] = useState<number>();
    const [imgHeight, setImgHeight] = useState<number>();
    const [lineprofileState, setLineprofileState] = useState<LineProfileState>(new LineProfileState(false, [], new RelativeLineProfile(new Coordinate(0, 0), new Coordinate(0, 1), new Coordinate(0, 0))));
    const [plotData, setPlotData] = React.useState([])

    const [openTable, setOpenTable] = React.useState<boolean>(false)
    const [imageInfo, setImageInfo] = React.useState<ImageInfo>(new ImageInfo(0, 0, 0))

    useEffect(() => {
        hubConnection.connect()
    }, [hubConnection]);

    const handleData = (input: any) => {
        let startTime = performance.now();
        let json = JSON.parse(input);
        setIntensities(json.data);
        setImgWidth(json.width);
        setImgHeight(Math.min(json.height, window.innerHeight));
        let endTime = performance.now();
        console.log(`Handling data took ${endTime - startTime} milliseconds`);
    };

    const [colormap, setColorMap] = React.useState("twilightShifted");
    const [jsonData, setJsonData] = React.useState({});
    const [isActive, setIsActive] = React.useState(false);

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

        hubConnection.requestProfiles(jsonConfig);
    }

    const throttled = useRef(throttle((data, lineprofiles) => sendLineprofileRequest(data, lineprofiles), 500));

    // useEffect(() => {
    //     jsonCallback(lineprofileState.lineProfiles.map(lp => {
    //         return {
    //             start: lp.start,
    //             end: lp.end
    //         }
    //     }), "lineprofiles")

    //     jsonCallback(imageInfo.id, "imageId")
    // }, [lineprofileState.lineProfiles])

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
        hubConnection.requestJob(jsonConfig);

    }, [jsonData, colormap]);

    useEffect(() => {
        let url = "/api/scatterstore/get?id=" + imageInfo.id + "&colormap=" + colormap;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => setRefIntensities(data));
    }, [imageInfo.id, colormap]);

    const sendJobInfo = () => {
        let jsonConfig = JSON.stringify({
            info: {
                body: JSON.stringify({
                    config: jsonData,
                    jobInfo: {
                        clientId: 0,
                        jobId: 0,
                        imageId: imageInfo.id,
                        intensityFormat: "doublePrecision",
                        simulationTargets: lineprofileState.lineProfiles
                    },
                    clientInfo: {
                        jobType: "fitting"
                    }
                }),
                history: []
            },
            userId: 0
        });
        console.log(jsonConfig)
        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfig
        };
        console.log(jsonConfig)
        let url = "/api/jobstore/push";
        fetch(url, requestOptions)
            .then(data => console.log(data));
    }

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
                        <LineProfileWrapper width={imgWidth} height={imgHeight} profileState={lineprofileState}
                            setProfileState={setLineprofileState}>
                            <ScatterImage intensities={intensities} width={imgWidth} height={imgHeight} />
                        </LineProfileWrapper>
                    </Box>
                    <Box
                        sx={{
                            paddingTop: 10,
                            paddingBottom: 10,
                            paddingLeft: 10
                        }}>
                        <Button onClick={() => sendJobInfo()}>
                            Create Job Description
                        </Button>
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
                        <LineProfileWrapper width={imgWidth} height={imgHeight} profileState={lineprofileState}
                            setProfileState={setLineprofileState}>
                            <ScatterImage intensities={refIntensities} width={imgWidth} height={imgHeight} />
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
                                <ImageTable setImageInfo={(updatedImageInfo: ImageInfo) => { console.log(updatedImageInfo); setImageInfo(updatedImageInfo) }} />
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
                                            <ColormapSelect colormap={colormap} setColormap={setColorMap} />
                                        </Grid>
                                        <Grid item xs={12} sm={12} md={12} lg={6} sx={{ justifyContent: "flex-end", display: 'flex' }}>
                                            <Button onClick={() => { setOpenTable(prevState => !prevState) }}>
                                                {openTable ? "Show graph" : "Show images"}
                                            </Button>
                                        </Grid>
                                    </Grid>
                                </Grid>

                                <Grid item xs={12} sm={7} md={7} lg={7}>
                                    <GisaxsShapes isSimulation={false} jsonCallback={jsonCallback} />
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