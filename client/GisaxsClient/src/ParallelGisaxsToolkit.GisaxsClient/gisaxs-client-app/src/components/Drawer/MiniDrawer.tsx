import { useState, forwardRef, useEffect } from "react";
import { styled, useTheme, Theme, CSSObject } from "@mui/material/styles";
import Box from "@mui/material/Box";
import MuiDrawer from "@mui/material/Drawer";
import MuiAppBar, { AppBarProps as MuiAppBarProps } from "@mui/material/AppBar";
import Toolbar from "@mui/material/Toolbar";
import List from "@mui/material/List";
import CssBaseline from "@mui/material/CssBaseline";
import Divider from "@mui/material/Divider";
import IconButton from "@mui/material/IconButton";
import MenuIcon from "@mui/icons-material/Menu";
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft";
import ChevronRightIcon from "@mui/icons-material/ChevronRight";
import DeviceHubIcon from "@mui/icons-material/DeviceHub";
import TimelineIcon from "@mui/icons-material/Timeline";
import Grid from "@mui/material/Grid";
import WorkHistoryIcon from '@mui/icons-material/WorkHistory';
import Login from "../Authentication/Login";
import { Link, LinkProps } from "react-router-dom";
import Register from "../Authentication/Register";
import ClickAwayComponent from "../Authentication/ClickAwayComponent";
import Logout from "../Authentication/Logout";
import DrawerLink from "./DrawerLink";

const drawerWidth = 240;

const openedMixin = (theme: Theme): CSSObject => ({
  width: drawerWidth,
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.enteringScreen,
  }),
  overflowX: "hidden",
});

const closedMixin = (theme: Theme): CSSObject => ({
  transition: theme.transitions.create("width", {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  overflowX: "hidden",
  width: `calc(${theme.spacing(7)} + 1px)`,
  [theme.breakpoints.up("sm")]: {
    width: `calc(${theme.spacing(8)} + 1px)`,
  },
});

const DrawerHeader = styled("div")(({ theme }) => ({
  display: "flex",
  alignItems: "center",
  justifyContent: "flex-end",
  padding: theme.spacing(0, 1),
  // necessary for content to be below app bar
  ...theme.mixins.toolbar,
}));

interface AppBarProps extends MuiAppBarProps {
  open?: boolean;
}

const AppBar = styled(MuiAppBar, {
  shouldForwardProp: (prop) => prop !== "open",
})<AppBarProps>(({ theme, open }) => ({
  zIndex: theme.zIndex.drawer + 1,
  transition: theme.transitions.create(["width", "margin"], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  ...(open && {
    marginLeft: drawerWidth,
    width: `calc(100% - ${drawerWidth}px)`,
    transition: theme.transitions.create(["width", "margin"], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
  }),
}));

const Drawer = styled(MuiDrawer, {
  shouldForwardProp: (prop) => prop !== "open",
})(({ theme, open }) => ({
  width: drawerWidth,
  flexShrink: 0,
  whiteSpace: "nowrap",
  boxSizing: "border-box",
  ...(open && {
    ...openedMixin(theme),
    "& .MuiDrawer-paper": openedMixin(theme),
  }),
  ...(!open && {
    ...closedMixin(theme),
    "& .MuiDrawer-paper": closedMixin(theme),
  }),
}));

export default function MiniDrawer() {
  const theme = useTheme();
  const [open, setOpen] = useState(false);
  const [authenticated, setAuthenticated] = useState(false)

  useEffect(() => {
    window.addEventListener("storage", () => {
      // When storage changes refetch
      let token = localStorage.getItem('apiToken')
      let isAuthenticated = token ? true : false
      setAuthenticated(isAuthenticated)
    });
    window.dispatchEvent(new Event("storage"));
    return () => {
      window.removeEventListener("storage", null);
    };
  }, []);



  const FittingLink = forwardRef<any, Omit<LinkProps, "to">>(
    (props, ref) => <Link ref={ref} to="/fitting" {...props} />
  );

  const SimulationLink = forwardRef<any, Omit<LinkProps, "to">>(
    (props, ref) => <Link ref={ref} to="/simulation" {...props} />
  );

  const JobsLink = forwardRef<any, Omit<LinkProps, "to">>(
    (props, ref) => <Link ref={ref} to="/jobs" {...props} />
  );

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar position="fixed" open={open}>
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            onClick={() => setOpen(true)}
            edge="start"
            sx={{
              marginRight: 5,
              ...(open && { display: "none" }),
            }}
          >
            <MenuIcon />
          </IconButton>
          {!authenticated ? (
            <Grid container justifyContent={"space-between"}>
              <Grid item xs={10} md={10} lg={10} />
              <Grid item xs={1} md={1} lg={1}>
                <ClickAwayComponent description="Login">
                  <Login />
                </ClickAwayComponent>
              </Grid>
              <Grid item xs={1} md={1} lg={1}>
                <ClickAwayComponent description="Register">
                  <Register />
                </ClickAwayComponent>
              </Grid>
            </Grid>
          ) : (
            <Grid container justifyContent={"space-between"}>
              <Grid item xs={10} md={10} lg={10} />
              <Grid item xs={1} md={1} lg={1}>
                <Logout />
              </Grid>
            </Grid>
          )
          }
        </Toolbar>
      </AppBar>

      <Drawer variant="permanent" open={open}>
        <DrawerHeader>
          <IconButton onClick={() => setOpen(false)}>
            {theme.direction === "rtl" ? (
              <ChevronRightIcon />
            ) : (
              <ChevronLeftIcon />
            )}
          </IconButton>
        </DrawerHeader>
        <Divider />
        <List>
          <DrawerLink description="Simulation" link={SimulationLink} open={open}>
            <DeviceHubIcon />
          </DrawerLink>

          <DrawerLink description="Fitting" link={FittingLink} open={open}>
            <TimelineIcon />
          </DrawerLink>

          <DrawerLink description="Jobs" link={JobsLink} open={open}>
            <WorkHistoryIcon />
          </DrawerLink>
        </List>
      </Drawer>
    </Box>
  );
}
