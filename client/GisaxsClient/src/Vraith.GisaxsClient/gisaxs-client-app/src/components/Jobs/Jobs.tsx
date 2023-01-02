import MiniDrawer from "../Drawer/MiniDrawer";
import CssBaseline from "@mui/material/CssBaseline"

import React, { useEffect, useMemo, useRef, useState } from "react";

const Jobs = () => {
    const [jobInfo, setJobInfo] = React.useState<JobInfo>(new JobInfo(0, { body: "{}" }))

    const receiveJobResult = (message: any) => {
        console.log(message)
    };

    const [hubConnection, _] = useState<MessageHubConnectionProvider>(
        new MessageHubConnectionProvider(
            `${localStorage.getItem("apiToken")}`,
            receiveJobResult,
            (message: string) => { },
            (message: string) => { }
        )
    )

    useEffect(() => {
        hubConnection.connect()
      }, [hubConnection]);

    const sendJob = () => {
        let jsonConfig = jobInfo.info.body;

        hubConnection.requestJob(jsonConfig, "");
    };

const Jobs = () => {
    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            
        </React.Fragment>
    );
};

export default Jobs