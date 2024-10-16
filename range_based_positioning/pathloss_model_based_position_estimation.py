from range_based_estimator import *


class PLPositioningEngine(RangeBasedEstimator):
    def __init__(self):
        super().__init__()
    
    def packet_perser(self):
        pass
    
    def gw_cord_collector(self):
        pass

    def link_budget_param_collector(self):
        pass

    def get_average_building_height(lat1, lon1, lat2, lon2):
        # Define the Overpass API query
        query = f"""
        [out:json];
        (
        way["building"]["height"]({lat1},{lon1},{lat2},{lon2});
        );
        out body;
        """
        
        # Send the request to the Overpass API
        response = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
        data = response.json()
        
        # Extract building heights
        heights = []
        for element in data['elements']:
            if 'tags' in element and 'height' in element['tags']:
                h = element['tags']['height']
                # The height elements are strings and may contain the units like "150 m"
                # So with the help of regular expression check we only take the value
                match = re.match(r"([-+]?\d*\.\d+|[-+]?\d+)", h)
                height = match.group(0)
                heights.append(float(height))
        
        # Calculate average height
        if heights:
            average_height = sum(heights) / len(heights)
            return average_height
        else:
            return None  # No building height data available


    def weather_info_collector(self):
        pass

    def haversine_distance(self):
        pass

    def range_predictor(self):
        pass

    def multilateration(self):
        pass


    