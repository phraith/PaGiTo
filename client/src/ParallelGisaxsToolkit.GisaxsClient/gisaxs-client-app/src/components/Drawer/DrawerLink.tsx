import ListItem from "@mui/material/ListItem/ListItem";
import ListItemIcon from "@mui/material/ListItemIcon/ListItemIcon";
import ListItemText from "@mui/material/ListItemText/ListItemText";
import Typography from "@mui/material/Typography/Typography";

interface DrawerLinkProps {
    children: React.ReactNode;
    description: string;
    link: any;
    open: boolean;
}

const DrawerLink: React.FC<DrawerLinkProps> = (props: DrawerLinkProps) => {
    return (
        <ListItem
            component={props.link}
            sx={{
                minHeight: 48,
                justifyContent: props.open ? "initial" : "center",
                px: 2.5,
            }}>
            <ListItemText
                primary={<Typography  sx={{ color: 'text.primary' }}>{props.description}</Typography>}
                sx={{ opacity: props.open ? 1 : 0 }} />
            <ListItemIcon
                sx={{
                    minWidth: 0,
                    mr: props.open ? 3 : "auto",
                    justifyContent: "center",
                }}>
                {props.children}
            </ListItemIcon>
        </ListItem>
    )
}

export default DrawerLink