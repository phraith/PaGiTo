import MiniDrawer from "../Drawer/MiniDrawer";
import CssBaseline from "@mui/material/CssBaseline"

import React, { useEffect, useMemo, useRef, useState } from "react";
import Grid from "@mui/material/Grid/Grid";
import JobsTable from "./JobsTable";
import { JobInfo } from "../../utility/JobInfo";
import { JsonViewer, NamedColorspace } from '@textea/json-viewer'
import Box from "@mui/material/Box/Box";
import Paper from "@mui/material/Paper/Paper";

export const ocean: NamedColorspace = {
    scheme: 'Ocean',
    author: 'Chris Kempson (http://chriskempson.com)',
    base00: '#2b303b',
    base01: '#343d46',
    base02: '#4f5b66',
    base03: '#65737e',
    base04: '#a7adba',
    base05: '#c0c5ce',
    base06: '#dfe1e8',
    base07: '#eff1f5',
    base08: '#bf616a',
    base09: '#d08770',
    base0A: '#ebcb8b',
    base0B: '#a3be8c',
    base0C: '#96b5b4',
    base0D: '#8fa1b3',
    base0E: '#b48ead',
    base0F: '#ab7967'
}

const Jobs = () => {
    const [jobInfo, setJobInfo] = React.useState<JobInfo>(new JobInfo(0, { body: "{}" }))
    return (
        <React.Fragment>
            <CssBaseline />
            <MiniDrawer />
            <Grid container spacing={2}>
                <Grid item xs={12} sm={12} md={12} lg={8}>
                    <Box display="flex" sx={{ flexDirection: "column", padding: 10 }}>
                        <Box display="flex" sx={{ paddingBottom: 1, width: "100%" }}>
                            <JobsTable setJobsInfo={(updatedJobInfo: JobInfo) => { console.log(updatedJobInfo); setJobInfo(updatedJobInfo) }} />
                            <Box sx={{ height: 500, overflow: 'auto', paddingLeft: 5, width: "50%" }}>
                                <JsonViewer value={JSON.parse(jobInfo.info.body)} />
                            </Box>
                        </Box>
                    </Box>
                </Grid>
            </Grid>
        </React.Fragment >
    );
};

export default Jobs