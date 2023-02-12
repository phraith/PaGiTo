import MiniDrawer from "../Drawer/MiniDrawer";
import Box from "@mui/material/Box"
import CssBaseline from "@mui/material/CssBaseline"
import Grid from "@mui/material/Grid"
import throttle from "lodash/throttle";
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
    const [currentInfoPath, setCurrentInfoPath] = useState<string>();
    const [imgWidth, setImgWidth] = useState<number>();
    const [imgHeight, setImgHeight] = useState<number>();
    const [lineprofileState, setLineprofileState] = useState<LineProfileState>(new LineProfileState(LineMode.Horizontal, [], new LineProfile(new Coordinate(0, 0), new Coordinate(0, 1))));
    const [simulatedPlotData, setSimulatedPlotData] = React.useState([])
    const [realPlotData, setRealPlotData] = React.useState([])

    const [openTable, setOpenTable] = React.useState<boolean>(false)
    const [imageInfo, setImageInfo] = React.useState<ImageInfo>(new ImageInfo(0, 0, 0))

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
    const [isActive, setIsActive] = React.useState(false);

    const jsonCallback = (value, key) => {
        jsonData[key] = value;
        setJsonData({ ...jsonData });
    };

    const throttled = useRef(throttle((jsonConfigForSimulation, jsonConfigForRealImage) => {

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
                let values = data.modifiedData
                let k = values.map((x: number, index: number) => { return [index, x]})
                traces.push(k)
                console.log(traces)
                setRealPlotData(traces[0])
            })


    }, 50));

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

        throttled.current(jsonConfigForSimulation, jsonConfigForRealImage)
    }, [lineprofileState?.currentLineProfile, jsonData]);

    useEffect(() => {
        jsonCallback(lineprofileState.lineProfiles.map(lp => {
            return {
                start: lp.start,
                end: lp.end
            }
        }), "lineprofiles")

        jsonCallback(imageInfo.id, "imageId")
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
            .then((data) => setRefIntensities(data.imageAsBase64));
    }, [imageInfo.id, colormap]);

    const sendJobInfo = () => {
        let jsonConfig = JSON.stringify({
            info: {
                body: JSON.stringify({
                    config: jsonData,
                    properties: {
                        imageId: imageInfo.id,
                        intensityFormat: "doublePrecision",
                        simulationTargets: lineprofileState.lineProfiles
                    },
                    meta: {
                        type: "fitting"
                    }
                }),
                history: []
            },
            userId: 0
        });

        const requestOptions = {
            method: 'POST',
            headers: {
                Authorization: `Bearer ${localStorage.getItem("apiToken")}`,
                Accept: "application/json",
                'Content-Type': 'application/json'
            },
            body: jsonConfig
        };
        const url = "/api/jobs";
        fetch(url, requestOptions)
            .then(data => console.log(data));
    }

    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2}>
                <Grid item xs={6} sm={6} md={6} lg={8}>
                    <Box sx={{ height: "100%", width: "100%", display: "flex", gap: 10, paddingTop: 10, paddingLeft: 10 }}>
                        <ScatterImageWithLineprofile width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={setLineprofileState} intensities={refIntensities} />
                        <ScatterImageWithLineprofile width={imgWidth} height={imgHeight} profileState={lineprofileState} setProfileState={setLineprofileState} intensities={intensities} />
                    </Box>
                </Grid>
                <Grid item xs={12} sm={12} md={12} lg={4}>
                    <Box display="flex" sx={{ flexDirection: "column", gap: 2, padding: 10 }}>

                        {!openTable &&
                            <Box sx={{ height: "100%", width: "100%" }}>
                                <LineProfileGraph simulatedData={simulatedPlotData} realData={realPlotData}/>
                            </Box>
                        }
                        {openTable &&
                            <Box sx={{ width: "100%", height: "100%" }}>
                                <ImageTable setImageInfo={(updatedImageInfo: ImageInfo) => { console.log(updatedImageInfo); setImageInfo(updatedImageInfo) }} />
                            </Box>
                        }
                        <Box display="flex" sx={{ paddingBottom: 1, gap: 2 }}>
                            <Instrumentation jsonCallback={jsonCallback} initialResX={imageInfo.width} initialResY={imageInfo.height} />
                            <UnitcellMeta jsonCallback={jsonCallback} />
                        </Box>
                        <Box display="flex" sx={{ paddingBottom: 1 }}>
                            <ColormapSelect colormap={colormap} setColormap={setColorMap} />
                            <Button onClick={() => sendJobInfo()}>
                                Create Job Description
                            </Button>
                            <Button onClick={() => { setOpenTable(prevState => !prevState) }}>
                                {openTable ? "Show graph" : "Show images"}
                            </Button>
                        </Box>
                        <Grid container spacing={2}>
                            <Grid item xs={7} sm={7} md={7} lg={7}>
                                <GisaxsShapes isSimulation={false} jsonCallback={jsonCallback} />
                            </Grid>
                            <Grid item xs={5} sm={5} md={5} lg={5}>
                                <Sample jsonCallback={jsonCallback} />
                            </Grid>
                        </Grid>
                    </Box>
                </Grid>
            </Grid>
        </React.Fragment>
    );
};

export default Fitting