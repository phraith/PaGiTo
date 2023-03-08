import MiniDrawer from "../Drawer/MiniDrawer";
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import Grid from "@mui/material/Grid"
import _debounce from "lodash/debounce"
import GisaxsShapes from "../GisaxsShapes/GisaxsShapes";
import Instrumentation from "../Instrumentation/Instrumentation";
import UnitcellMeta from "../UnitcellMeta/UnitcellMeta";
import { useEffect, useRef, useState } from "react";
import * as React from "react";
import Sample from "../Sample/Sample";
import { Coordinate, LineMode, LineProfile, LineProfileState } from "../../utility/LineProfile";
import ImageTable from "./ImageTable";
import { Button } from "@mui/material";
import { ImageInfo } from "../../utility/ImageInfo";
import { MessageHubConnectionProvider } from "../../utility/MessageHubConnectionProvider";
import ColormapSelect from "../Colormap";
import ScatterImageWithLineprofile from "../ScatterImage/ScatterImageWithLineprofile";
import LineProfileGraph from "./LineProfileGraphECharts";

const Fitting = () => {
    const getLineprofiles = (hash: any) => {
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
            .then((data) => {
                console.log(data)
                let json = JSON.parse(data.response)
                let traces = []
                let values = json.numericResults[0].modifiedData
                let k = values.map((x: number, index: number) => { return [index, x] })
                traces.push(k)
                setSimulatedPlotData(traces[0])
            })
    }

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
                ["receiveJobResult", (message: string) => receiveJobResult(message)],
                ["receiveLineProfile", (message: string) => getLineprofiles(message)]
            ]
        )
    )

    const [intensities, setIntensities] = useState<string>();
    const [refIntensities, setRefIntensities] = useState<string>();
    const [imgWidth, setImgWidth] = useState<number>();
    const [imgHeight, setImgHeight] = useState<number>();
    const [lineprofileState, setLineprofileState] = useState<LineProfileState>(new LineProfileState(LineMode.Horizontal, [], new LineProfile(new Coordinate(0, 0), new Coordinate(0, 1))));
    const [simulatedPlotData, setSimulatedPlotData] = React.useState([])
    const [realPlotData, setRealPlotData] = React.useState([])

    const [openTable, setOpenTable] = React.useState<boolean>(false)
    const [imageInfo, setImageInfo] = React.useState<ImageInfo>(new ImageInfo(1, 0, 0))

    useEffect(() => {
        hubConnection.connect()
    }, [hubConnection]);

    const handleData = (input: any) => {
        let startTime = performance.now();
        let json = JSON.parse(input.response);
        setIntensities(json.jpegResults[0].data);
        setImgWidth(json.jpegResults[0].width);
        setImgHeight(json.jpegResults[0].height);
        let endTime = performance.now();
        console.log(`Handling data took ${endTime - startTime} milliseconds`);
    };

    const [colormap, setColorMap] = React.useState("twilightShifted");
    const [jsonData, setJsonData] = React.useState({});

    const jsonCallback = (value, key) => {
        jsonData[key] = value;
        setJsonData({ ...jsonData });
    };


    const requestLineProfiles = (jsonConfigForSimulation, jsonConfigForRealImage) => {
        const requestOptions1 = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(
                {
                    "jsonConfig": jsonConfigForSimulation
                }
            )
        };

        let url1 = "/api/job";
        fetch(url1, requestOptions1)
            .then(data => console.log(data));


        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfigForRealImage
        };
        const url = `/api/image/profile`;
        fetch(url, requestOptions)
            .then((response) => response.json())
            .then((data) => {
                let traces = []
                console.log(data)
                let values = data.modifiedData
                let k = values.map((x: number, index: number) => { return [index, x] })
                traces.push(k)
                console.log(traces)
                setRealPlotData(traces[0])
            })
    }


    const debounced = useRef(_debounce((jsonConfigForSimulation, jsonConfigForRealImage) => {
        requestLineProfiles(jsonConfigForSimulation, jsonConfigForRealImage)
    }, 50))

    // const throttled = useRef(throttle((jsonConfigForSimulation, jsonConfigForRealImage) => {
    //     requestLineProfiles(jsonConfigForSimulation, jsonConfigForRealImage)
    // }, 50));

    useEffect(() => {
        let jsonConfigForSimulation = JSON.stringify({
            meta: {
                type: "simulation",
                notification: "receiveLineProfile"
            },
            properties: {
                intensityFormat: "doublePrecision",
                simulationTargets: [
                    lineprofileState?.currentLineProfile.inverseHeight(imgHeight)
                ]
            },
            config: {
                ...jsonData,
            },
        });

        let jsonConfigForRealImage = JSON.stringify(
            {
                target: {
                    id: imageInfo.id,
                    target: lineprofileState?.currentLineProfile.inverseHeight(imgHeight)
                }
            })

        debounced.current(jsonConfigForSimulation, jsonConfigForRealImage)
    }, [lineprofileState?.currentLineProfile, jsonData]);

    useEffect(() => {
        jsonCallback(lineprofileState.lineProfiles.map(lp => {
            return {
                start: lp.start,
                end: lp.end
            }
        }), "lineprofiles")
    }, [lineprofileState.lineProfiles])

    useEffect(() => {
        let jsonConfig = JSON.stringify({
            meta: {
                type: "simulation",
                notification: "receiveJobResult",
                persist: true,
                execute: false,
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

    useEffect(() => {
        const url = `/api/image/${imageInfo.id}/${colormap}`;
        fetch(url, {
            method: "GET",
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
            },
        })
            .then((response) => response.json())
            .then((data) => {
                setRefIntensities(data.imageAsBase64)
                setImgWidth(data.width);
                setImgHeight(data.height);
                console.log("asdasd")
                console.log(data)
            }
            );
    }, [imageInfo.id, colormap]);

    const sendJobInfo = () => {
        let jsonConfig = JSON.stringify({
            meta: {
                type: "fitting",
                notification: "receiveJobResult",
                persist: true,
                execute: false,
                colormap: colormap
            },
            properties: {
                intensityFormat: "doublePrecision",
                imageId: imageInfo.id,
                simulationTargets: lineprofileState.lineProfiles
            },
            config: {
                ...jsonData,
            },
        });

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
    }

    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2} direction={"row"} sx={{ padding: 10 }}>
                <Grid item xs={4} sm={4} md={4} lg={4}>
                    <ScatterImageWithLineprofile width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={setLineprofileState} intensities={refIntensities} />
                </Grid>
                <Grid item xs={4} sm={6} md={4} lg={4}>
                    <ScatterImageWithLineprofile width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={setLineprofileState} intensities={intensities} />
                </Grid>
                <Grid item xs={4} sm={4} md={4} lg={4}>
                    <Box display="flex" flexDirection={"column"} sx={{ gap: 2 }}>
                        <Box display="flex" sx={{ gap: 2, height: "30vh" }}>
                            {!openTable
                                ? <LineProfileGraph simulatedData={simulatedPlotData} realData={realPlotData} />
                                : <ImageTable setImageInfo={(updatedImageInfo: ImageInfo) => { setImageInfo(updatedImageInfo) }} />
                            }
                        </Box>

                        <Box display="flex" sx={{ gap: 2 }}>
                            <Instrumentation jsonCallback={jsonCallback} initialResX={imageInfo.width} initialResY={imageInfo.height} />
                            <UnitcellMeta jsonCallback={jsonCallback} />
                        </Box>

                        <Box display="flex" sx={{ gap: 2 }}>
                            <ColormapSelect colormap={colormap} setColormap={setColorMap} />
                            <Button variant="outlined" onClick={() => sendJobInfo()}>
                                Create
                            </Button>
                            <Button variant="outlined" onClick={() => { setOpenTable(prevState => !prevState) }}>
                                {openTable ? "Graph" : "Images"}
                            </Button>
                        </Box>

                        <Grid container spacing={2} sx={{ height: "30vh" }}>
                            <Grid item xs={7} sm={7} md={7} lg={7} sx={{ height: "100%" }}>
                                <GisaxsShapes isSimulation={false} jsonCallback={jsonCallback} />
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

export default Fitting