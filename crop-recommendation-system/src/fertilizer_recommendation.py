"""
Fertilizer Recommendation Module
Provides intelligent fertilizer recommendations based on soil NPK levels and crop requirements
"""

import pandas as pd
from config import CROP_INFO, FERTILIZER_DATABASE


class FertilizerRecommender:
    """
    Recommends fertilizers based on current soil NPK and crop requirements
    """
    
    def __init__(self):
        """
        Initialize the fertilizer recommender
        """
        self.crop_info = CROP_INFO
        self.fertilizer_db = FERTILIZER_DATABASE
    
    def get_crop_requirements(self, crop_name):
        """
        Get optimal NPK requirements for a crop
        
        Args:
            crop_name (str): Name of the crop
            
        Returns:
            dict: Optimal NPK values
        """
        crop_name_lower = crop_name.lower()
        
        if crop_name_lower in self.crop_info:
            return self.crop_info[crop_name_lower]['optimal_npk']
        else:
            print(f"Warning: Crop '{crop_name}' not found in database")
            return None
    
    def calculate_npk_deficit(self, current_npk, optimal_npk):
        """
        Calculate the deficit/surplus of NPK
        
        Args:
            current_npk (dict): Current soil NPK levels
            optimal_npk (dict): Optimal NPK requirements
            
        Returns:
            dict: NPK deficit (positive means need to add, negative means surplus)
        """
        deficit = {
            'N': optimal_npk['N'] - current_npk['N'],
            'P': optimal_npk['P'] - current_npk['P'],
            'K': optimal_npk['K'] - current_npk['K']
        }
        
        return deficit
    
    def recommend_fertilizer(self, current_npk, crop_name, area_hectares=1.0):
        """
        Recommend fertilizer based on current NPK and crop requirements
        
        Args:
            current_npk (dict): Current soil NPK levels {'N': x, 'P': y, 'K': z}
            crop_name (str): Name of the recommended crop
            area_hectares (float): Area of land in hectares
            
        Returns:
            dict: Fertilizer recommendations
        """
        # Get optimal NPK for the crop
        optimal_npk = self.get_crop_requirements(crop_name)
        
        if optimal_npk is None:
            return {
                'status': 'error',
                'message': f'Crop {crop_name} not found in database'
            }
        
        # Calculate deficit
        deficit = self.calculate_npk_deficit(current_npk, optimal_npk)
        
        # Determine fertilizer strategy
        recommendations = []
        total_cost = 0
        
        # Check each nutrient
        for nutrient in ['N', 'P', 'K']:
            if deficit[nutrient] > 5:  # Significant deficit
                # Find best fertilizer for this nutrient
                best_fertilizer = self._find_best_fertilizer(nutrient, deficit[nutrient], area_hectares)
                if best_fertilizer:
                    recommendations.append(best_fertilizer)
                    total_cost += best_fertilizer['cost']
        
        # If multiple deficits, also suggest NPK complex fertilizer
        deficit_count = sum(1 for v in deficit.values() if v > 5)
        if deficit_count >= 2:
            npk_recommendation = self._recommend_npk_complex(deficit, area_hectares)
            if npk_recommendation:
                recommendations.append({
                    'type': 'alternative',
                    'fertilizer': npk_recommendation['fertilizer'],
                    'quantity_kg': npk_recommendation['quantity_kg'],
                    'cost': npk_recommendation['cost'],
                    'reason': 'Balanced NPK complex for multiple deficiencies'
                })
        
        return {
            'status': 'success',
            'crop': crop_name,
            'current_npk': current_npk,
            'optimal_npk': optimal_npk,
            'deficit': deficit,
            'recommendations': recommendations,
            'total_cost': total_cost,
            'area_hectares': area_hectares
        }
    
    def _find_best_fertilizer(self, nutrient, deficit_amount, area_hectares):
        """
        Find the best fertilizer for a specific nutrient deficit
        
        Args:
            nutrient (str): 'N', 'P', or 'K'
            deficit_amount (float): Amount of deficit
            area_hectares (float): Area in hectares
            
        Returns:
            dict: Fertilizer recommendation
        """
        # Filter fertilizers that contain the required nutrient
        suitable_fertilizers = {
            name: data for name, data in self.fertilizer_db.items()
            if data[nutrient] > 0
        }
        
        if not suitable_fertilizers:
            return None
        
        # Find most cost-effective fertilizer
        best_fertilizer = None
        best_cost_per_unit = float('inf')
        
        for fert_name, fert_data in suitable_fertilizers.items():
            cost_per_unit_nutrient = fert_data['cost_per_kg'] / fert_data[nutrient]
            if cost_per_unit_nutrient < best_cost_per_unit:
                best_cost_per_unit = cost_per_unit_nutrient
                best_fertilizer = fert_name
        
        # Calculate required quantity
        fert_data = self.fertilizer_db[best_fertilizer]
        nutrient_percentage = fert_data[nutrient] / 100
        required_kg = (deficit_amount * area_hectares) / nutrient_percentage
        cost = required_kg * fert_data['cost_per_kg']
        
        return {
            'nutrient': nutrient,
            'fertilizer': best_fertilizer,
            'quantity_kg': round(required_kg, 2),
            'cost': round(cost, 2),
            'nutrient_content': f"{fert_data[nutrient]}% {nutrient}",
            'reason': f'To address {nutrient} deficit of {deficit_amount} units'
        }
    
    def _recommend_npk_complex(self, deficit, area_hectares):
        """
        Recommend NPK complex fertilizer for multiple deficiencies
        
        Args:
            deficit (dict): NPK deficit
            area_hectares (float): Area in hectares
            
        Returns:
            dict: NPK complex recommendation
        """
        # Find NPK complex fertilizers
        npk_fertilizers = {
            name: data for name, data in self.fertilizer_db.items()
            if 'NPK' in name
        }
        
        if not npk_fertilizers:
            return None
        
        # Choose NPK 20:20:20 as balanced option
        if 'NPK 20:20:20' in npk_fertilizers:
            fert_name = 'NPK 20:20:20'
        else:
            fert_name = list(npk_fertilizers.keys())[0]
        
        fert_data = self.fertilizer_db[fert_name]
        
        # Calculate based on highest deficit
        max_deficit = max(abs(deficit['N']), abs(deficit['P']), abs(deficit['K']))
        avg_npk_content = (fert_data['N'] + fert_data['P'] + fert_data['K']) / 3
        required_kg = (max_deficit * area_hectares * 3) / avg_npk_content
        cost = required_kg * fert_data['cost_per_kg']
        
        return {
            'fertilizer': fert_name,
            'quantity_kg': round(required_kg, 2),
            'cost': round(cost, 2)
        }
    
    def get_fertilizer_application_guide(self, fertilizer_name):
        """
        Get application guidelines for a fertilizer
        
        Args:
            fertilizer_name (str): Name of the fertilizer
            
        Returns:
            dict: Application guidelines
        """
        guidelines = {
            'Urea': {
                'application_time': 'Split application - 50% at sowing, 25% at tillering, 25% at flowering',
                'method': 'Broadcast or band placement',
                'precautions': 'Avoid application during hot weather, water immediately after application'
            },
            'DAP': {
                'application_time': 'At sowing or planting',
                'method': 'Band placement near seed',
                'precautions': 'Do not mix with urea, maintain gap between seed and fertilizer'
            },
            'MOP': {
                'application_time': 'At sowing or as basal dose',
                'method': 'Broadcast and incorporate into soil',
                'precautions': 'Apply in moist soil, avoid chloride-sensitive crops'
            },
            'NPK 20:20:20': {
                'application_time': 'At sowing and as top dressing',
                'method': 'Broadcast or fertigation',
                'precautions': 'Ensure uniform distribution, water after application'
            }
        }
        
        return guidelines.get(fertilizer_name, {
            'application_time': 'Follow local agricultural guidelines',
            'method': 'As per crop requirements',
            'precautions': 'Consult agricultural expert'
        })
    
    def format_recommendation_report(self, recommendation):
        """
        Format recommendation as a readable report
        
        Args:
            recommendation (dict): Recommendation from recommend_fertilizer()
            
        Returns:
            str: Formatted report
        """
        if recommendation['status'] == 'error':
            return recommendation['message']
        
        report = []
        report.append("="*60)
        report.append("FERTILIZER RECOMMENDATION REPORT")
        report.append("="*60)
        report.append(f"\nCrop: {recommendation['crop'].title()}")
        report.append(f"Area: {recommendation['area_hectares']} hectares")
        
        report.append("\n--- Current Soil Status ---")
        report.append(f"Nitrogen (N):    {recommendation['current_npk']['N']} units")
        report.append(f"Phosphorous (P): {recommendation['current_npk']['P']} units")
        report.append(f"Potassium (K):   {recommendation['current_npk']['K']} units")
        
        report.append("\n--- Optimal Requirements ---")
        report.append(f"Nitrogen (N):    {recommendation['optimal_npk']['N']} units")
        report.append(f"Phosphorous (P): {recommendation['optimal_npk']['P']} units")
        report.append(f"Potassium (K):   {recommendation['optimal_npk']['K']} units")
        
        report.append("\n--- Nutrient Deficit/Surplus ---")
        for nutrient, value in recommendation['deficit'].items():
            status = "DEFICIT" if value > 0 else "SURPLUS" if value < 0 else "BALANCED"
            report.append(f"{nutrient}: {abs(value):.1f} units ({status})")
        
        if recommendation['recommendations']:
            report.append("\n--- Recommended Fertilizers ---")
            for i, rec in enumerate(recommendation['recommendations'], 1):
                if rec.get('type') != 'alternative':
                    report.append(f"\n{i}. {rec['fertilizer']}")
                    report.append(f"   Quantity: {rec['quantity_kg']} kg")
                    report.append(f"   Cost: ₹{rec['cost']:.2f}")
                    report.append(f"   Purpose: {rec['reason']}")
            
            # Alternative recommendations
            alternatives = [r for r in recommendation['recommendations'] if r.get('type') == 'alternative']
            if alternatives:
                report.append("\n--- Alternative Option ---")
                for alt in alternatives:
                    report.append(f"\n{alt['fertilizer']}")
                    report.append(f"   Quantity: {alt['quantity_kg']} kg")
                    report.append(f"   Cost: ₹{alt['cost']:.2f}")
                    report.append(f"   Note: {alt['reason']}")
            
            report.append(f"\n--- Total Estimated Cost ---")
            report.append(f"₹{recommendation['total_cost']:.2f}")
        else:
            report.append("\n✓ Soil nutrient levels are adequate for this crop!")
            report.append("No additional fertilizer required at this time.")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


def main():
    """
    Demonstration of fertilizer recommendation
    """
    recommender = FertilizerRecommender()
    
    # Example 1: Rice cultivation
    print("\nExample 1: Rice Cultivation")
    current_soil = {'N': 40, 'P': 30, 'K': 25}
    recommendation = recommender.recommend_fertilizer(current_soil, 'rice', area_hectares=2.0)
    print(recommender.format_recommendation_report(recommendation))
    
    # Example 2: Wheat cultivation
    print("\n\nExample 2: Wheat Cultivation")
    current_soil = {'N': 60, 'P': 50, 'K': 40}
    recommendation = recommender.recommend_fertilizer(current_soil, 'wheat', area_hectares=1.5)
    print(recommender.format_recommendation_report(recommendation))


if __name__ == "__main__":
    main()
